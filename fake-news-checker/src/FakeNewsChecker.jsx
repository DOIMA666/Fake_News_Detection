import React, { useState, useEffect, useCallback } from 'react';
import { Shield, Sparkles, Zap, BarChart3, Search, Clock } from 'lucide-react';

import Dashboard from './components/Dashboard';
import Checker from './components/Checker';
import History from './components/History';

const FakeNewsChecker = () => {
  const [inputType, setInputType] = useState('text');
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);
  const [showResult, setShowResult] = useState(false);
  const [activeTab, setActiveTab] = useState('checker');
  const [historySubTab, setHistorySubTab] = useState('personal'); 
  const [communityHistory, setCommunityHistory] = useState([]);
  const [history, setHistory] = useState([]);
  const [savedItems, setSavedItems] = useState([]);
                
  const API_URL = import.meta.env.VITE_API_URL ;

  const [stats, setStats] = useState({
    totalChecks: 0,
    trueNews: 0,
    falseNews: 0,
    uncertain: 0,
    todayChecks: 0,
    accuracy: 0
  });

  const [recentChecks, setRecentChecks] = useState([]);
  const [trendingTopics, setTrendingTopics] = useState([]);
  const [dataLoading, setDataLoading] = useState(false);

  useEffect(() => {
    const savedHistory = localStorage.getItem('checkHistory');
    if (savedHistory) {
      setHistory(JSON.parse(savedHistory));
    }
    const saved = localStorage.getItem('savedItems');
    if (saved) {
      setSavedItems(JSON.parse(saved));
    }
  }, []);

  const loadDashboardData = useCallback(async () => {
    setDataLoading(true);
    try {
      const statsRes = await fetch(`${API_URL}/api/stats`);
      const statsData = await statsRes.json();
      if (statsData.success) {
        setStats(statsData.data);
      }

      const recentRes = await fetch(`${API_URL}/api/recent?limit=10`);
      const recentData = await recentRes.json();
      if (recentData.success) {
        setRecentChecks(recentData.data);
      }

      const trendingRes = await fetch(`${API_URL}/api/trending?limit=5`);
      const trendingData = await trendingRes.json();
      if (trendingData.success) {
        setTrendingTopics(trendingData.data);
      }
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
      setStats({
        totalChecks: 0, trueNews: 0, falseNews: 0, uncertain: 0, todayChecks: 0, accuracy: 0
      });
      setRecentChecks([]);
      setTrendingTopics([]);
    } finally {
      setDataLoading(false);
    }
  }, [API_URL]); 

  const loadCommunityHistory = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/api/history?skip=0&limit=20`);
      const data = await response.json();
      if (data.success) {
        setCommunityHistory(data.data);
      }
    } catch (err) {
      console.error('Failed to load community history:', err);
      setCommunityHistory([]);
    }
  }, [API_URL]);

  useEffect(() => {
    if (activeTab === 'dashboard') {
      loadDashboardData();
    }
  }, [activeTab, loadDashboardData]); 

  useEffect(() => {
    if (activeTab === 'history' && historySubTab === 'community') {
      loadCommunityHistory();
    }
  }, [activeTab, historySubTab, loadCommunityHistory]); 

  useEffect(() => {
    if (loading) {
      const interval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 90) return prev;
          return prev + Math.random() * 10;
        });
      }, 500);
      return () => clearInterval(interval);
    } else {
      setProgress(0);
    }
  }, [loading]);

  useEffect(() => {
    if (result) {
      setTimeout(() => setShowResult(true), 100);
    } else {
      setShowResult(false);
    }
  }, [result]);

  const handleSubmit = async () => {
    if (!content.trim()) {
      setError('Vui lòng nhập nội dung cần kiểm tra');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setProgress(0);

    try {
      const response = await fetch(`${API_URL}/api/check`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content: content,
          input_type: inputType,
          num_sources: 15
        })
      });

      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        const text = await response.text();
        console.error('Non-JSON response:', text);
        throw new Error('Server trả về response không hợp lệ. Vui lòng kiểm tra server logs.');
      }

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || data.detail || 'Có lỗi xảy ra');
      }

      if (data.success) {
        setProgress(100);
        setTimeout(() => {
          setResult(data);
          const newHistoryItem = {
            id: Date.now(),
            content: content.substring(0, 100) + '...',
            verdict: data.verdict,
            timestamp: new Date().toISOString(),
            type: inputType
          };
          const updatedHistory = [newHistoryItem, ...history].slice(0, 50);
          setHistory(updatedHistory);
          localStorage.setItem('checkHistory', JSON.stringify(updatedHistory));
        }, 300);
      } else {
        setError(data.message || 'Không thể xử lý yêu cầu');
      }

    } catch (err) {
      let errorMessage = err.message;
      
      if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
        errorMessage = 'Không thể kết nối đến server. Vui lòng kiểm tra:\n• Kết nối mạng của bạn\n• Server đã được khởi động chưa\n• Địa chỉ API có chính xác không';
      } else if (err.message.includes('JSON')) {
        errorMessage = 'Lỗi parse dữ liệu từ server. Kiểm tra console để biết thêm chi tiết.';
      }
      
      console.error('Error details:', err);
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleSubmit();
    } else if (e.key === 'Enter' && !e.shiftKey && inputType === 'url') {
      e.preventDefault();
      handleSubmit();
    }
  };

  const saveResult = () => {
    if (!result) return;
    const savedItem = {
      id: Date.now(),
      content: content.substring(0, 100) + '...',
      result: result,
      timestamp: new Date().toISOString()
    };
    const updated = [savedItem, ...savedItems];
    setSavedItems(updated);
    localStorage.setItem('savedItems', JSON.stringify(updated));
    alert('Đã lưu kết quả!');
  };

  const shareResult = () => {
    if (!result) return;
    const shareText = `Kết quả kiểm tra tin: ${result.verdict.label}\nĐộ tin cậy: ${result.verdict.confidence_percentage}%`;
    if (navigator.share) {
      navigator.share({
        title: 'Kết quả kiểm tra tin giả',
        text: shareText,
      });
    } else {
      navigator.clipboard.writeText(shareText);
      alert('Đã copy kết quả vào clipboard!');
    }
  };

  const exportResult = () => {
    if (!result) return;

    // 1. Tạo nội dung
    const title = "KẾT QUẢ KIỂM TRA TIN GIẢ";
    const separator = "=".repeat(50);
    const now = new Date();
    const timeDisplay = now.toLocaleString('vi-VN');
    
    let contentText = `${title}\n${separator}\n`;
    contentText += `Thời gian kiểm tra: ${timeDisplay}\n`;
    contentText += `Nội dung: "${content}"\n\n`;
    
    contentText += `>>> KẾT LUẬN: ${result.verdict.label.toUpperCase()}\n`;
    contentText += `>>> ĐỘ TIN CẬY: ${Math.round(result.verdict.confidence_percentage)}%\n`;
    contentText += `>>> CHỦ ĐỀ: ${result.processed_data?.category_label || 'Không xác định'}\n`;
    contentText += `>>> GIẢI THÍCH: ${result.verdict.explanation}\n\n`;
    
    if (result.voting_summary) {
      contentText += `PHÂN TÍCH CHI TIẾT\n${separator}\n`;
      contentText += `- Ủng hộ/Xác nhận: ${result.voting_summary.support_count} nguồn\n`;
      contentText += `- Bác bỏ/Phủ định: ${result.voting_summary.refute_count} nguồn\n`;
      contentText += `- Bàn luận/Trung lập: ${result.voting_summary.discuss_count} nguồn\n\n`;
    }

    if (result.references && result.references.length > 0) {
      contentText += `NGUỒN TIN THAM KHẢO (${result.references.length})\n${separator}\n`;
      result.references.forEach((ref, index) => {
        contentText += `${index + 1}. [${ref.stance.label}] ${ref.title}\n`;
        contentText += `   Nguồn: ${ref.domain} (Tier ${ref.credibility.tier})\n`;
        contentText += `   Link: ${ref.url}\n\n`;
      });
    }
    contentText += `${separator}\nĐược tạo bởi hệ thống Fake News Checker.`;

    const dateStr = `${now.getDate()}-${now.getMonth() + 1}-${now.getFullYear()}`;
    const timeStr = `${now.getHours()}h${now.getMinutes()}`;
    
    const fileName = `Ket_qua_kiem_tra_tin_gia-${dateStr}_${timeStr}.txt`;

    const dataBlob = new Blob([contentText], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const deleteSavedItem = (id) => {
    const updated = savedItems.filter(item => item.id !== id);
    setSavedItems(updated);
    localStorage.setItem('savedItems', JSON.stringify(updated));
  };

  const handleHistorySelect = (item, isCommunity) => {

    const contentText = isCommunity ? item.content_preview : item.content;
    const inputTypeVal = isCommunity ? item.input_type : item.type;
    
    setContent(contentText || '');
    setInputType(inputTypeVal || 'text');

    let resultData;

    if (item.result) {
        resultData = item.result;
    } 

    else if (isCommunity) {
        resultData = {
            db_id: item.id,
            verdict: {
                code: item.verdict_code,
                label: item.verdict_label,
                explanation: "Chi tiết không khả dụng (Dữ liệu cũ hoặc chưa tải đầy đủ).",
                confidence_percentage: item.confidence_percentage,
            },
            processed_data: { category: 'other', category_label: 'Đã lưu' },
            references: [], 
            voting_summary: null
        };
    } else {

        resultData = item.result || {
             verdict: item.verdict,

        };
    }

    setResult(resultData);
    setShowResult(true);
    
    setActiveTab('checker');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 relative overflow-hidden">
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-10 w-72 h-72 bg-blue-200 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-blob"></div>
        <div className="absolute top-40 right-10 w-72 h-72 bg-purple-200 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-blob animation-delay-2000"></div>
        <div className="absolute -bottom-32 left-1/3 w-72 h-72 bg-pink-200 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-blob animation-delay-4000"></div>
      </div>

      <style>{`
        @keyframes blob {
          0%, 100% { transform: translate(0, 0) scale(1); }
          33% { transform: translate(30px, -50px) scale(1.1); }
          66% { transform: translate(-20px, 20px) scale(0.9); }
        }
        .animate-blob { animation: blob 8s infinite; }
        .animation-delay-2000 { animation-delay: 2s; }
        .animation-delay-4000 { animation-delay: 4s; }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in { animation: fadeIn 0.6s ease-out; }
      `}</style>

      <div className="max-w-7xl mx-auto px-4 py-8 relative z-10">
        <div className="text-center mb-8 md:mb-12">
          <div className="inline-flex items-center justify-center w-16 h-16 md:w-20 md:h-20 bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-600 rounded-2xl shadow-2xl mb-4 md:mb-6 transform hover:rotate-6 transition-transform">
            <Shield className="w-8 h-8 md:w-10 md:h-10 text-white" />
          </div>
          <h1 className="text-3xl md:text-6xl font-extrabold bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 bg-clip-text text-transparent mb-2 md:mb-3">
            Kiểm Tra Tin Giả
          </h1>
          <p className="text-sm md:text-lg text-gray-600 flex items-center justify-center gap-2">
            <Sparkles className="w-4 h-4 md:w-5 md:h-5 text-indigo-500" />
            Phát hiện tin giả bằng AI
            <Zap className="w-4 h-4 md:w-5 md:h-5 text-amber-500" />
          </p>
        </div>

        <div className="flex justify-center mb-6 md:mb-10">
          <div className="inline-flex bg-white rounded-2xl shadow-lg p-1.5 md:p-2 border border-gray-200 max-w-full overflow-x-auto">
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`flex items-center gap-1.5 md:gap-2 px-3 md:px-6 py-2 md:py-3 rounded-xl font-semibold text-xs md:text-base transition-all duration-300 whitespace-nowrap ${
                activeTab === 'dashboard'
                  ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white shadow-lg'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              }`}
            >
              <BarChart3 className="w-4 h-4 md:w-5 md:h-5" />
              Dashboard
            </button>
            <button
              onClick={() => setActiveTab('checker')}
              className={`flex items-center gap-1.5 md:gap-2 px-3 md:px-6 py-2 md:py-3 rounded-xl font-semibold text-xs md:text-base transition-all duration-300 whitespace-nowrap ${
                activeTab === 'checker'
                  ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white shadow-lg'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              }`}
            >
              <Search className="w-4 h-4 md:w-5 md:h-5" />
              Kiểm tra
            </button>
            <button
              onClick={() => setActiveTab('history')}
              className={`flex items-center gap-1.5 md:gap-2 px-3 md:px-6 py-2 md:py-3 rounded-xl font-semibold text-xs md:text-base transition-all duration-300 whitespace-nowrap ${
                activeTab === 'history'
                  ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white shadow-lg'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              }`}
            >
              <Clock className="w-4 h-4 md:w-5 md:h-5" />
              Lịch sử
            </button>
          </div>
        </div>

        {activeTab === 'dashboard' && (
          <Dashboard 
            dataLoading={dataLoading}
            stats={stats}
            recentChecks={recentChecks}
            trendingTopics={trendingTopics}
            setActiveTab={setActiveTab}
          />
        )}
        
        {activeTab === 'checker' && (
          <Checker 
            inputType={inputType}
            setInputType={setInputType}
            content={content}
            setContent={setContent}
            loading={loading}
            progress={progress}
            handleSubmit={handleSubmit}
            handleKeyPress={handleKeyPress}
            error={error}
            result={result}
            showResult={showResult}
            saveResult={saveResult}
            shareResult={shareResult}
            exportResult={exportResult}
            apiURL={API_URL}
          />
        )}

        {activeTab === 'history' && (
          <History 
            historySubTab={historySubTab}
            setHistorySubTab={setHistorySubTab}
            history={history}
            setHistory={setHistory}
            communityHistory={communityHistory}
            loadCommunityHistory={loadCommunityHistory}
            savedItems={savedItems}
            deleteSavedItem={deleteSavedItem}
            onSelect={handleHistorySelect}
          />
        )}

        <div className="text-center mt-16 space-y-3">
          <div className="flex items-center justify-center gap-3">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-full flex items-center justify-center">
              <Sparkles className="w-6 h-6 text-white" />
            </div>
            <p className="text-xl font-bold text-gray-700">Hệ thống AI phân tích thông minh</p>
          </div>
          <p className="text-sm text-gray-500 font-medium">Kết quả mang tính chất tham khảo</p>
        </div>
      </div>
    </div>
  );
};

export default FakeNewsChecker;