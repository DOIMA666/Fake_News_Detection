
import React, { useState, useEffect } from 'react';
import { AlertCircle, CheckCircle, Search, ExternalLink, Globe, FileText, Sparkles, Shield, TrendingUp, Info, Zap, Award } from 'lucide-react';

const FakeNewsChecker = () => {
  const [inputType, setInputType] = useState('text');
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);
  const [showResult, setShowResult] = useState(false);

  
  //const API_URL = 'https://Nhom05TLCN-Fake-News-detection.hf.space';
  const API_URL = 'http://localhost:8000';

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
          num_sources: 5
        })
      });

      const data = await response.json();

      if (!response.ok) {
        // Xử lý lỗi theo mã HTTP
        let errorMessage = 'Có lỗi xảy ra';
        
        if (response.status === 404) {
          errorMessage = 'Không tìm thấy API. Vui lòng kiểm tra lại đường dẫn server.';
        } else if (response.status === 422) {
          errorMessage = 'Dữ liệu không hợp lệ. Vui lòng kiểm tra lại nội dung hoặc URL bạn nhập.';
        } else if (response.status === 500) {
          errorMessage = 'Lỗi server. Vui lòng thử lại sau ít phút.';
        } else if (response.status === 503) {
          errorMessage = 'Server đang bảo trì. Vui lòng thử lại sau.';
        } else if (response.status === 403) {
          errorMessage = 'Không có quyền truy cập. Vui lòng kiểm tra cấu hình API.';
        } else if (response.status === 401) {
          errorMessage = 'Chưa xác thực. Vui lòng kiểm tra API key hoặc token.';
        } else if (response.status === 400) {
          errorMessage = data.detail || 'Yêu cầu không hợp lệ. Vui lòng kiểm tra lại dữ liệu đầu vào.';
        } else {
          errorMessage = data.detail || errorMessage;
        }
        
        throw new Error(errorMessage);
      }

      if (data.success) {
        setProgress(100);
        setTimeout(() => setResult(data), 300);
      } else {
        setError(data.message || 'Không thể xử lý yêu cầu');
      }

    } catch (err) {
      // Xử lý lỗi kết nối
      let errorMessage = err.message;
      
      if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
        errorMessage = 'Không thể kết nối đến server. Vui lòng kiểm tra:\n• Kết nối mạng của bạn\n• Server đã được khởi động chưa\n• Địa chỉ API có chính xác không';
      } else if (err.message.includes('timeout') || err.message.includes('timed out')) {
        errorMessage = 'Kết nối bị timeout. Vui lòng kiểm tra kết nối mạng và thử lại.';
      }
      
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    // Ctrl/Cmd + Enter để submit cho textarea
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleSubmit();
    }
    // Chỉ Enter (không có Shift) cho input URL
    else if (e.key === 'Enter' && !e.shiftKey && inputType === 'url') {
      e.preventDefault();
      handleSubmit();
    }
  };

  const getVerdictConfig = (verdict) => {
    const configs = {
      'HIGHLY_LIKELY_TRUE': {
        bg: 'from-emerald-500 via-green-500 to-teal-500',
        border: 'border-emerald-400',
        text: 'text-emerald-900',
        icon: CheckCircle,
        glow: 'shadow-emerald-500/50',
        badge: 'Tin cậy cao'
      },
      'LIKELY_TRUE': {
        bg: 'from-green-400 via-emerald-400 to-teal-400',
        border: 'border-green-300',
        text: 'text-green-800',
        icon: CheckCircle,
        glow: 'shadow-green-400/50',
        badge: 'Có thể tin'
      },
      'UNCERTAIN': {
        bg: 'from-amber-400 via-yellow-400 to-orange-400',
        border: 'border-amber-300',
        text: 'text-amber-900',
        icon: Info,
        glow: 'shadow-amber-400/50',
        badge: 'Chưa rõ ràng'
      },
      'LIKELY_FALSE': {
        bg: 'from-orange-500 via-red-500 to-pink-500',
        border: 'border-orange-400',
        text: 'text-orange-900',
        icon: AlertCircle,
        glow: 'shadow-orange-500/50',
        badge: 'Nghi ngờ'
      },
      'HIGHLY_LIKELY_FALSE': {
        bg: 'from-red-600 via-rose-600 to-pink-600',
        border: 'border-red-500',
        text: 'text-red-900',
        icon: AlertCircle,
        glow: 'shadow-red-500/50',
        badge: 'Có vấn đề'
      }
    };
    return configs[verdict] || configs['UNCERTAIN'];
  };

  const CircularProgress = ({ value, label }) => {
    const radius = 45;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (value / 100) * circumference;

    return (
      <div className="flex flex-col items-center justify-center">
        <div className="relative inline-flex items-center justify-center">
          <svg className="transform -rotate-90 w-36 h-36">
            {/* Background circle - màu nhạt */}
            <circle
              cx="72"
              cy="72"
              r={radius}
              stroke="rgba(255, 255, 255, 0.25)"
              strokeWidth="8"
              fill="none"
            />
            {/* Progress circle - màu đậm với gradient */}
            <circle
              cx="72"
              cy="72"
              r={radius}
              stroke="url(#progressGradient)"
              strokeWidth="8"
              fill="none"
              strokeDasharray={circumference}
              strokeDashoffset={offset}
              className="transition-all duration-1000 ease-out"
              strokeLinecap="round"
              style={{ filter: 'drop-shadow(0 0 8px rgba(255, 255, 255, 0.6))' }}
            />
            {/* Gradient definition */}
            <defs>
              <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="rgba(255, 255, 255, 0.95)" />
                <stop offset="100%" stopColor="rgba(200, 240, 255, 1)" />
              </linearGradient>
            </defs>
          </svg>
          <div className="absolute flex flex-col items-center">
            <span className="text-4xl font-bold text-white drop-shadow-lg">{Math.round(value)}%</span>
          </div>
        </div>
        <span className="text-sm font-semibold text-white/90 mt-3">{label}</span>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 py-12 px-4 relative overflow-hidden">
      {/* Animated background elements */}
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
        .animate-blob {
          animation: blob 8s infinite;
        }
        .animation-delay-2000 {
          animation-delay: 2s;
        }
        .animation-delay-4000 {
          animation-delay: 4s;
        }
        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-12px); }
        }
        .animate-float {
          animation: float 3s ease-in-out infinite;
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in {
          animation: fadeIn 0.6s ease-out;
        }
      `}</style>

      <div className="max-w-6xl mx-auto relative z-10">
        {/* Header */}
        <div className="text-center mb-16 animate-float">
          <div className="inline-flex items-center justify-center w-24 h-24 bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-600 rounded-3xl shadow-2xl mb-8 transform hover:scale-110 hover:rotate-3 transition-all duration-300">
            <Shield className="w-12 h-12 text-white" />
          </div>
          <h1 className="text-6xl md:text-7xl font-extrabold bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 bg-clip-text text-transparent mb-4 tracking-tight">
            Kiểm Tra Tin Giả
          </h1>
          <p className="text-xl text-gray-600 flex items-center justify-center gap-2 font-medium">
            <Sparkles className="w-5 h-5 text-indigo-500" />
            Phát hiện tin giả bằng AI thông minh
            <Zap className="w-5 h-5 text-amber-500" />
          </p>
        </div>

        {/* Main Card */}
        <div className="bg-white rounded-3xl shadow-xl p-10 mb-10 border border-gray-100 transform transition-all duration-500 hover:shadow-2xl">
          {/* Type Selector */}
          <div className="flex gap-4 mb-10">
            <button
              onClick={() => setInputType('text')}
              className={`flex-1 py-5 px-8 rounded-2xl font-semibold text-base transition-all duration-300 transform hover:scale-105 ${
                inputType === 'text'
                  ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white shadow-lg shadow-blue-500/50'
                  : 'bg-white text-gray-700 hover:bg-gray-100 border-2 border-gray-200'
              }`}
            >
              <FileText className="inline-block w-6 h-6 mr-3" />
              Nhập văn bản
            </button>
            <button
              onClick={() => setInputType('url')}
              className={`flex-1 py-5 px-8 rounded-2xl font-semibold text-base transition-all duration-300 transform hover:scale-105 ${
                inputType === 'url'
                  ? 'bg-gradient-to-r from-purple-500 to-pink-600 text-white shadow-lg shadow-purple-500/50'
                  : 'bg-white text-gray-700 hover:bg-gray-100 border-2 border-gray-200'
              }`}
            >
              <Globe className="inline-block w-6 h-6 mr-3" />
              Nhập URL
            </button>
          </div>

          {/* Input Area */}
          <div className="mb-8">
            <label className="block text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
              {inputType === 'text' ? (
                <>
                  <span className="text-2xl">🧾</span>
                  <span>Nội dung cần kiểm tra</span>
                </>
              ) : (
                <>
                  <span className="text-2xl">🔗</span>
                  <span>URL bài báo</span>
                </>
              )}
            </label>
            {inputType === 'text' ? (
              <>
                <textarea
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  onKeyDown={handleKeyPress}
                  placeholder="Dán nội dung tin tức cần kiểm tra vào đây..."
                  className="w-full h-52 p-5 border-2 border-gray-200 rounded-2xl focus:ring-4 focus:ring-blue-500/30 focus:border-blue-500 transition-all duration-300 resize-none text-gray-800 text-base leading-relaxed"
                  disabled={loading}
                />
                <div className="mt-5 p-5 bg-gradient-to-r from-blue-50 to-indigo-50 border-l-4 border-blue-400 rounded-xl">
                  <div className="flex items-start gap-3">
                    <div className="flex-shrink-0 w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center">
                      <Sparkles className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <p className="text-sm font-bold text-blue-900 mb-2">Mẹo để có kết quả chính xác nhất</p>
                      <ul className="text-sm text-blue-800 space-y-2">
                        <li className="flex items-start gap-2">
                          <span className="text-blue-500 font-bold mt-0.5">→</span>
                          <span>Nội dung cần có nghĩa và đầy đủ thông tin</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <span className="text-blue-500 font-bold mt-0.5">→</span>
                          <span>Càng nhiều nội dung, độ chính xác càng cao</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <span className="text-blue-500 font-bold mt-0.5">→</span>
                          <span>Nên nhập toàn bộ nội dung bài báo thay vì chỉ tiêu đề</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <span className="text-blue-500 font-bold mt-0.5">→</span>
                          <span>Đây là công cụ dùng để kiểm tra tính chính xác của tin tức, không phải công cụ tìm kiếm tin tức</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <span className="text-blue-500 font-bold mt-0.5">⌨️</span>
                          <span><kbd className="px-2 py-1 bg-white rounded border border-blue-300 text-xs font-mono">Ctrl</kbd> + <kbd className="px-2 py-1 bg-white rounded border border-blue-300 text-xs font-mono">Enter</kbd> để kiểm tra nhanh</span>
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <>
                <input
                  type="url"
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  onKeyDown={handleKeyPress}
                  placeholder="https://vnexpress.net/tieu-de-bai-bao-12345.html"
                  className="w-full p-5 border-2 border-gray-200 rounded-2xl focus:ring-4 focus:ring-purple-500/30 focus:border-purple-500 transition-all duration-300 text-gray-800 text-base"
                  disabled={loading}
                />
                <div className="mt-5 p-5 bg-gradient-to-r from-purple-50 to-pink-50 border-l-4 border-purple-400 rounded-xl">
                  <div className="flex items-start gap-3">
                    <div className="flex-shrink-0 w-10 h-10 bg-purple-500 rounded-full flex items-center justify-center">
                      <Info className="w-5 h-5 text-white" />
                    </div>
                    <div className="flex-1">
                      <p className="text-sm font-bold text-purple-900 mb-2">Lưu ý quan trọng</p>
                      <p className="text-sm text-purple-800 mb-3">
                        Nhập URL của <span className="font-bold">bài báo cụ thể</span>, không phải trang chủ hay chủ đề
                      </p>
                      <div className="text-sm bg-white/60 p-4 rounded-lg space-y-2 border border-purple-200">
                        <div className="flex items-center gap-2 text-green-700">
                          <CheckCircle className="w-4 h-4" />
                          <span className="font-medium">https://vnexpress.net/tieu-de-bai-viet-12345.html</span>
                        </div>
                        <div className="flex items-center gap-2 text-red-700">
                          <AlertCircle className="w-4 h-4" />
                          <span className="font-medium">https://vnexpress.net/topic/covid-19</span>
                        </div>
                        <div className="flex items-center gap-2 text-purple-600 mt-3 pt-3 border-t border-purple-200">
                          <Info className="w-4 h-4" />
                          <span className="text-xs">Nhấn <kbd className="px-1.5 py-0.5 bg-white rounded border border-purple-300 font-mono">Enter</kbd> để kiểm tra nhanh</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
          
          {/* Trusted Sources Info */}
          <div className="mb-8 p-5 bg-gradient-to-r from-emerald-50 to-teal-50 border-l-4 border-emerald-400 rounded-xl">
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 w-10 h-10 bg-emerald-500 rounded-full flex items-center justify-center">
                <Shield className="w-5 h-5 text-white" />
              </div>
              <div>
                <p className="text-sm font-bold text-emerald-900 mb-2">Nguồn kiểm chứng uy tín</p>
                <p className="text-sm text-emerald-800">
                  Hệ thống kiểm tra với <span className="font-bold">5 trang báo hàng đầu Việt Nam</span>: VnExpress, Thanh Niên, Dân Trí, Tuổi Trẻ, VietnamNet
                </p>
              </div>
            </div>
          </div>

          {/* Submit Button */}
          <button
            onClick={handleSubmit}
            disabled={loading || !content.trim()}
            className="w-full bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white py-6 px-10 rounded-2xl font-bold text-lg hover:from-blue-700 hover:via-indigo-700 hover:to-purple-700 disabled:from-gray-300 disabled:via-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed transition-all duration-300 flex items-center justify-center gap-4 shadow-xl hover:shadow-2xl transform hover:scale-[1.02] active:scale-95 relative overflow-hidden group"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/20 to-white/0 transform translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700"></div>
            {loading ? (
              <>
                <div className="animate-spin rounded-full h-6 w-6 border-3 border-white border-t-transparent"></div>
                <span>Đang phân tích...</span>
              </>
            ) : (
              <>
                <Search className="w-6 h-6" />
                <span>Kiểm tra ngay</span>
                <Zap className="w-5 h-5" />
              </>
            )}
          </button>

          {/* Progress Bar */}
          {loading && (
            <div className="mt-8 animate-fade-in">
              <div className="h-3 bg-gray-100 rounded-full overflow-hidden shadow-inner">
                <div 
                  className="h-full bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-600 transition-all duration-500 ease-out relative overflow-hidden"
                  style={{ width: `${progress}%` }}
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-pulse"></div>
                </div>
              </div>
              <p className="text-center text-sm text-gray-600 mt-3 font-medium">
                {progress < 30 ? 'Đang phân tích nội dung...' : 
                 progress < 60 ? 'Đang tìm kiếm nguồn tin...' : 
                 progress < 90 ? 'Đang so sánh với các nguồn uy tín...' : 
                 'Gần hoàn thành...'}
              </p>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="mt-8 p-6 bg-gradient-to-r from-red-50 to-pink-50 border-l-4 border-red-400 rounded-xl flex items-start gap-4 animate-fade-in shadow-lg">
              <div className="flex-shrink-0 w-10 h-10 bg-red-500 rounded-full flex items-center justify-center">
                <AlertCircle className="w-5 h-5 text-white" />
              </div>
              <div className="flex-1">
                <p className="font-bold text-red-900 text-lg mb-1">Đã xảy ra lỗi</p>
                <p className="text-red-700 text-sm whitespace-pre-line">{error}</p>
              </div>
            </div>
          )}
        </div>

        {/* Result Card */}
        {result && result.verdict && (
          <div className={`transition-all duration-700 transform ${showResult ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
            <div className="bg-white rounded-3xl shadow-xl p-10 border border-gray-100">
              {/* Verdict Header */}
              <div className={`relative p-10 rounded-3xl mb-10 bg-gradient-to-br ${getVerdictConfig(result.verdict.code).bg} shadow-2xl ${getVerdictConfig(result.verdict.code).glow} overflow-hidden`}>
                <div className="absolute inset-0 bg-black/5 backdrop-blur-sm"></div>
                
                {/* Decorative elements */}
                <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full blur-3xl"></div>
                <div className="absolute bottom-0 left-0 w-48 h-48 bg-white/10 rounded-full blur-2xl"></div>
                
                <div className="relative z-10">
                  <div className="flex items-center justify-between mb-8">
                    <div className="flex items-center gap-4">
                      <h2 className="text-4xl font-extrabold text-white drop-shadow-lg">Kết quả kiểm tra</h2>
                      <span className="px-4 py-2 bg-white/20 backdrop-blur-sm rounded-full text-sm font-bold text-white border border-white/30">
                        {getVerdictConfig(result.verdict.code).badge}
                      </span>
                    </div>
                    {React.createElement(getVerdictConfig(result.verdict.code).icon, {
                      className: "w-14 h-14 text-white drop-shadow-lg animate-pulse"
                    })}
                  </div>
                  
                  <p className="text-3xl font-bold text-white mb-4 drop-shadow-md">{result.verdict.label}</p>
                  <p className="text-white/95 text-lg leading-relaxed drop-shadow-sm">{result.verdict.explanation}</p>
                  
                  {/* Stats */}
                  <div className="grid grid-cols-2 gap-8 mt-10">
                    <div className="bg-white/15 backdrop-blur-md p-6 rounded-2xl border-2 border-white/30 transform hover:scale-105 transition-transform duration-300 shadow-xl">
                      <CircularProgress 
                        value={result.verdict.similarity_percentage} 
                        label="Độ tương đồng"
                      />
                    </div>
                    <div className="bg-white/15 backdrop-blur-md p-6 rounded-2xl border-2 border-white/30 transform hover:scale-105 transition-transform duration-300 shadow-xl">
                      <CircularProgress 
                        value={result.verdict.confidence_percentage} 
                        label="Độ tin cậy"
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Keywords */}
              {result.keywords && result.keywords.length > 0 && (
                <div className="mb-10 animate-fade-in">
                  <h3 className="font-bold text-gray-900 mb-5 text-2xl flex items-center gap-3">
                    <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center">
                      <TrendingUp className="w-5 h-5 text-white" />
                    </div>
                    <span>Từ khóa chính</span>
                  </h3>
                  <div className="flex flex-wrap gap-3">
                    {result.keywords.map((keyword, idx) => (
                      <span 
                        key={idx} 
                        className="px-5 py-3 bg-gradient-to-r from-blue-50 to-indigo-50 text-blue-700 rounded-xl text-sm font-semibold border-2 border-blue-200 transform hover:scale-110 hover:-translate-y-1 transition-all duration-300 cursor-default shadow-md hover:shadow-xl"
                        style={{
                          animationDelay: `${idx * 100}ms`
                        }}
                      >
                        #{keyword}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* References */}
              {result.references && result.references.length > 0 && (
                <div className="animate-fade-in">
                  <h3 className="font-bold text-gray-900 mb-6 text-2xl flex items-center gap-3">
                    <div className="w-10 h-10 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl flex items-center justify-center">
                      <Award className="w-5 h-5 text-white" />
                    </div>
                    <span>Nguồn tin uy tín</span>
                  </h3>
                  <div className="space-y-4">
                    {result.references.map((ref, idx) => (
                      <div 
                        key={idx} 
                        className="group p-6 border-2 border-gray-100 rounded-2xl hover:border-blue-300 hover:shadow-xl transition-all duration-300 bg-gradient-to-r from-white to-blue-50/30 transform hover:-translate-y-1"
                        style={{
                          animationDelay: `${idx * 150}ms`
                        }}
                      >
                        <div className="flex items-start justify-between gap-6">
                          <div className="flex-1">
                            <a 
                              href={ref.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-blue-600 hover:text-blue-800 font-bold mb-2 block text-lg group-hover:underline leading-relaxed capitalize"
                            >
                              {ref.title || ref.url}
                            </a>
                            <p className="text-sm text-gray-600 font-medium flex items-center gap-2">
                              <Globe className="w-4 h-4" />
                              {ref.domain}
                            </p>
                          </div>
                          <div className="flex items-center gap-4 flex-shrink-0">
                            <div className="text-center bg-gradient-to-br from-blue-50 to-indigo-50 px-5 py-3 rounded-xl border-2 border-blue-200">
                              <div className="text-3xl font-extrabold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                                {ref.similarity_percentage}%
                              </div>
                              <div className="text-xs text-gray-600 font-semibold mt-1">tương đồng</div>
                            </div>
                            <ExternalLink className="w-6 h-6 text-gray-400 group-hover:text-blue-600 transition-colors" />
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Footer */}
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