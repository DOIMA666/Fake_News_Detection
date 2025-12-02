import React, { useState } from 'react';
import { Bookmark, Share2, Download, Target, CheckCircle, AlertCircle, Info, Shield, Globe, ExternalLink, ThumbsUp, ThumbsDown, MessageSquare, XCircle } from 'lucide-react';

// Mock helper functions
const getVerdictConfig = (code) => {
  const configs = {
    'LIKELY_TRUE': {
      icon: CheckCircle,
      label: 'Thông Tin Có Khả Năng Đúng',
      color: 'text-green-600',
      bg: 'from-green-500 via-green-600 to-emerald-600',
      bgLight: 'bg-green-50',
      badge: 'bg-green-100 text-green-700'
    },
    'LIKELY_FALSE': {
      icon: XCircle,
      label: 'Thông Tin Có Khả Năng Sai',
      color: 'text-red-600',
      bg: 'from-red-500 via-rose-500 to-pink-600',
      bgLight: 'bg-red-50',
      badge: 'bg-red-100 text-red-700'
    },
    'UNCERTAIN': {
      icon: AlertCircle,
      label: 'Thông Tin Không Chắc Chắn',
      color: 'text-yellow-600',
      bg: 'from-yellow-500 via-amber-500 to-orange-600',
      bgLight: 'bg-yellow-50',
      badge: 'bg-yellow-100 text-yellow-700'
    }
  };
  return configs[code] || configs['UNCERTAIN'];
};

const getCategoryInfo = (category) => {
  const categories = {
    'politics': { icon: '🏛️', label: 'Chính trị', color: 'border-purple-300 bg-purple-50' },
    'health': { icon: '🏥', label: 'Y tế', color: 'border-green-300 bg-green-50' },
    'crime': { icon: '⚖️', label: 'Pháp luật', color: 'border-red-300 bg-red-50' },
    'entertainment': { icon: '🎭', label: 'Giải trí', color: 'border-pink-300 bg-pink-50' },
    'sports': { icon: '⚽', label: 'Thể thao', color: 'border-blue-300 bg-blue-50' },
    'economy': { icon: '💰', label: 'Kinh tế', color: 'border-yellow-300 bg-yellow-50' },
    'technology': { icon: '💻', label: 'Công nghệ', color: 'border-indigo-300 bg-indigo-50' },
    'other': { icon: '📰', label: 'Khác', color: 'border-gray-300 bg-gray-50' }
  };
  return categories[category] || categories['other'];
};

const getStanceConfig = (code) => {
  const configs = {
    'support': {
      icon: CheckCircle,
      badge: 'bg-green-100 text-green-700 border-green-200'
    },
    'refute': {
      icon: XCircle,
      badge: 'bg-red-100 text-red-700 border-red-200'
    },
    'discuss': {
      icon: Info,
      badge: 'bg-yellow-100 text-yellow-700 border-yellow-200'
    },
    'unrelated': {
      icon: AlertCircle,
      badge: 'bg-gray-100 text-gray-700 border-gray-200'
    }
  };
  // Handle case-insensitive matching safely
  const safeCode = (code || '').toLowerCase();
  return configs[safeCode] || configs['unrelated'];
};

const CheckerResult = ({ result, showResult, saveResult, shareResult, exportResult, apiURL }) => {
  
  
  const getFeedbackKey = () => (result && result.db_id) ? `feedback_${result.db_id}` : null;
  
  const getInitialFeedbackStatus = () => {
    try {
      const key = getFeedbackKey();
      if (!key) return null;
      return localStorage.getItem(key) || null;
    } catch  {
      return null;
    }
  };
  
  const [feedbackStatus, setFeedbackStatus] = useState(getInitialFeedbackStatus);
  
  // Nếu không có kết quả thì không render gì cả (tránh lỗi render bên dưới)
  if (!result || !result.verdict) return null;

  const verdictConfig = getVerdictConfig(result.verdict.code);
  const VerdictIcon = verdictConfig.icon;

  const handleFeedback = async (isCorrect) => {
    if (!result.db_id) {
      setFeedbackStatus('submitted');
      return;
    }

    setFeedbackStatus('submitting');
    try {
      
      const baseUrl = apiURL || 'http://localhost:8000';
      
      await fetch(`${baseUrl}/api/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          check_id: result.db_id,
          is_correct: isCorrect
        })
      });
      
      // Save feedback status to localStorage
      const key = getFeedbackKey();
      if (key) {
        localStorage.setItem(key, 'submitted');
      }
      
      setTimeout(() => setFeedbackStatus('submitted'), 500);
    } catch (err) {
      console.error("Lỗi gửi feedback:", err);
      // Vẫn lưu state để UI không bị treo
      const key = getFeedbackKey();
      if (key) {
        localStorage.setItem(key, 'submitted');
      }
      setFeedbackStatus('submitted');
    }
  };

  return (
    <div className={`transition-all duration-700 transform ${showResult ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
      <div className="bg-white rounded-3xl shadow-xl p-8 md:p-10 border border-gray-100">
        
        {/* ===== VERDICT HEADER CARD (SIMPLIFIED) ===== */}
        <div className={`relative p-8 md:p-10 rounded-3xl mb-8 bg-gradient-to-br ${verdictConfig.bg} shadow-2xl overflow-hidden`}>
          <div className="absolute inset-0 bg-black/5 backdrop-blur-sm"></div>
          <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full blur-3xl"></div>
          
          <div className="relative z-10">
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-6 gap-4">
              <h2 className="text-3xl md:text-4xl font-extrabold text-white drop-shadow-lg">
                Kết quả kiểm tra
              </h2>
              <div className="flex gap-2">
                <button 
                  onClick={saveResult} 
                  className="p-3 bg-white/20 hover:bg-white/30 rounded-xl backdrop-blur-sm transition-all hover:scale-110" 
                  title="Lưu kết quả"
                >
                  <Bookmark className="w-5 h-5 text-white" />
                </button>
                <button 
                  onClick={shareResult} 
                  className="p-3 bg-white/20 hover:bg-white/30 rounded-xl backdrop-blur-sm transition-all hover:scale-110" 
                  title="Chia sẻ"
                >
                  <Share2 className="w-5 h-5 text-white" />
                </button>
                <button 
                  onClick={exportResult} 
                  className="p-3 bg-white/20 hover:bg-white/30 rounded-xl backdrop-blur-sm transition-all hover:scale-110" 
                  title="Xuất dữ liệu"
                >
                  <Download className="w-5 h-5 text-white" />
                </button>
              </div>
            </div>

            {/* Center Content - Verdict Display */}
            <div className="flex flex-col items-center justify-center text-center space-y-6">
              <div className="w-32 h-32 md:w-40 md:h-40 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center shadow-2xl">
                <VerdictIcon className="w-16 h-16 md:w-20 md:h-20 text-white drop-shadow-lg" />
              </div>
              
              <div className="space-y-3">
                <p className="text-4xl md:text-5xl font-extrabold text-white drop-shadow-md">
                  {verdictConfig.label}
                </p>
                <p className="text-white/95 text-base md:text-lg leading-relaxed drop-shadow-sm max-w-2xl">
                  {result.verdict.explanation}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* FEEDBACK SECTION */}
        <div className="mb-8 animate-fade-in">
          {!feedbackStatus ? (
            <div className="p-5 md:p-6 bg-gray-50 rounded-2xl border border-dashed border-gray-300 flex flex-col md:flex-row items-center justify-between gap-4">
              <div className="flex items-center gap-3 text-gray-700">
                <div className="p-2 bg-white rounded-full shadow-sm">
                  <MessageSquare className="w-5 h-5 text-blue-500" />
                </div>
                <div>
                  <p className="font-bold text-base md:text-lg">Đánh giá kết quả này</p>
                  <p className="text-xs md:text-sm text-gray-500">Phản hồi của bạn giúp AI thông minh hơn</p>
                </div>
              </div>

              <div className="flex w-full md:w-auto gap-3">
                <button
                  onClick={() => handleFeedback(true)}
                  className="flex-1 md:flex-none flex items-center justify-center gap-2 px-5 py-3 bg-white border border-green-200 text-green-700 rounded-xl hover:bg-green-50 hover:border-green-300 hover:shadow-md transition-all duration-300"
                >
                  <ThumbsUp className="w-5 h-5" />
                  <span className="font-semibold">Chính xác</span>
                </button>
                <button
                  onClick={() => handleFeedback(false)}
                  className="flex-1 md:flex-none flex items-center justify-center gap-2 px-5 py-3 bg-white border border-red-200 text-red-700 rounded-xl hover:bg-red-50 hover:border-red-300 hover:shadow-md transition-all duration-300"
                >
                  <ThumbsDown className="w-5 h-5" />
                  <span className="font-semibold">Chưa đúng</span>
                </button>
              </div>
            </div>
          ) : feedbackStatus === 'submitting' ? (
             <div className="p-6 bg-gray-50 rounded-2xl border border-gray-100 flex justify-center items-center gap-3">
                <div className="animate-spin h-5 w-5 border-2 border-blue-500 border-t-transparent rounded-full"></div>
                <span className="text-gray-600 font-medium">Đang gửi phản hồi...</span>
             </div>
          ) : (
            <div className="p-4 md:p-5 bg-blue-50 text-blue-800 rounded-2xl border border-blue-100 flex flex-col md:flex-row items-center justify-center gap-3 text-center md:text-left">
              <div className="p-1 bg-blue-100 rounded-full">
                <CheckCircle className="w-6 h-6 text-blue-600" />
              </div>
              <div>
                <span className="font-bold">Cảm ơn đóng góp của bạn! </span>
                <span className="text-blue-700 text-sm md:text-base">Dữ liệu đã được ghi nhận để cải thiện độ chính xác của hệ thống.</span>
              </div>
            </div>
          )}
        </div>

        {/* ===== CATEGORY BADGE ===== */}
        {result.processed_data?.category && (
          <div className="mb-8 animate-fade-in">
            <h3 className="font-bold text-gray-900 mb-4 text-lg md:text-xl flex items-center gap-2">
              <Target className="w-5 h-5 text-purple-600" />
              Chủ đề
            </h3>
            {(() => {
              const catInfo = getCategoryInfo(result.processed_data.category);
              return (
                <div className={`inline-flex items-center gap-3 px-5 py-3 rounded-2xl border-2 ${catInfo.color} shadow-md hover:shadow-lg transition-shadow`}>
                  <span className="text-2xl md:text-3xl">{catInfo.icon}</span>
                  <span className="font-bold text-base md:text-lg uppercase tracking-wide">
                    {result.processed_data.category_label || catInfo.label}
                  </span>
                </div>
              );
            })()}
          </div>
        )}

        {/* ===== VOTING SUMMARY (SIMPLIFIED) ===== */}
        {result.voting_summary && (
          <div className="mb-10 p-6 bg-gradient-to-r from-purple-50 to-pink-50 rounded-2xl border-2 border-purple-200 animate-fade-in shadow-md">
            <h3 className="font-bold text-gray-900 mb-5 text-lg md:text-xl flex items-center gap-2">
              <Info className="w-6 h-6 text-purple-600" />
              Phân tích chi tiết
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Support Card */}
              <div className="bg-white p-6 rounded-xl border-2 border-green-200 text-center transform hover:scale-105 transition-all shadow-sm hover:shadow-md">
                <CheckCircle className="w-8 h-8 text-green-600 mx-auto mb-3" />
                <div className="text-4xl font-bold text-green-600 mb-2">
                  {result.voting_summary.support_count}
                </div>
                <div className="text-sm text-gray-600 font-medium">Bài báo xác nhận</div>
              </div>
              
              {/* Refute Card */}
              <div className="bg-white p-6 rounded-xl border-2 border-red-200 text-center transform hover:scale-105 transition-all shadow-sm hover:shadow-md">
                <XCircle className="w-8 h-8 text-red-600 mx-auto mb-3" />
                <div className="text-4xl font-bold text-red-600 mb-2">
                  {result.voting_summary.refute_count}
                </div>
                <div className="text-sm text-gray-600 font-medium">Bài báo bác bỏ</div>
              </div>
              
              {/* Discuss Card */}
              <div className="bg-white p-6 rounded-xl border-2 border-yellow-200 text-center transform hover:scale-105 transition-all shadow-sm hover:shadow-md">
                <Info className="w-8 h-8 text-yellow-600 mx-auto mb-3" />
                <div className="text-4xl font-bold text-yellow-600 mb-2">
                  {result.voting_summary.discuss_count}
                </div>
                <div className="text-sm text-gray-600 font-medium">Bài báo bàn luận</div>
              </div>
            </div>
          </div>
        )}

        {/* ===== KEYWORDS ===== */}
        {result.keywords && result.keywords.length > 0 && (
          <div className="mb-10 animate-fade-in">
            <h3 className="font-bold text-gray-900 mb-5 text-lg md:text-xl flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-md">
                <Target className="w-5 h-5 text-white" />
              </div>
              <span>Từ khóa chính</span>
            </h3>
            <div className="flex flex-wrap gap-3">
              {result.keywords.map((keyword, idx) => (
                <span 
                  key={idx} 
                  className="px-5 py-3 bg-gradient-to-r from-blue-50 to-indigo-50 text-blue-700 rounded-xl text-sm font-semibold border-2 border-blue-200 transform hover:scale-110 hover:-translate-y-1 transition-all duration-300 cursor-default shadow-sm hover:shadow-md"
                >
                  #{keyword}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* ===== REFERENCES (SIMPLIFIED) ===== */}
        {result.references && result.references.length > 0 && (
          <div className="animate-fade-in">
            <h3 className="font-bold text-gray-900 mb-6 text-lg md:text-xl flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl flex items-center justify-center shadow-md">
                <Shield className="w-5 h-5 text-white" />
              </div>
              <span>Nguồn tin tham khảo</span>
              <span className="text-sm font-normal text-gray-500">
                ({result.references.length} bài)
              </span>
            </h3>
            
            <div className="space-y-4">
              {result.references.map((ref, idx) => {
                const stanceConfig = getStanceConfig(ref.stance.code);
                const StanceIcon = stanceConfig.icon;
                
                const tierColors = {
                  1: 'border-green-300 bg-green-50',
                  2: 'border-blue-300 bg-blue-50',
                  3: 'border-orange-300 bg-orange-50',
                  4: 'border-gray-300 bg-gray-50'
                };
                
                return (
                  <div 
                    key={idx} 
                    className={`group p-6 border-2 rounded-2xl hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1 ${tierColors[ref.credibility.tier] || 'border-gray-200 bg-white'}`}
                  >
                    <div className="flex flex-col lg:flex-row items-start justify-between gap-4 lg:gap-6">
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-3 flex-wrap">
                          <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-bold border-2 ${stanceConfig.badge}`}>
                            <StanceIcon className="w-4 h-4" />
                            <span>{ref.stance.label}</span>
                            <span className="text-xs opacity-75">
                              ({ref.stance.confidence}%)
                            </span>
                          </span>
                          
                          <span 
                            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-bold text-white"
                            style={{ backgroundColor: ref.credibility.color }}
                          >
                            <Shield className="w-4 h-4" />
                            <span>Tier {ref.credibility.tier}</span>
                          </span>
                        </div>
                        
                        <a 
                          href={ref.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-gray-800 hover:text-blue-600 hover:underline font-bold mb-2 block text-lg leading-snug line-clamp-2"
                        >
                          {ref.title || ref.url}
                        </a>
                        
                        <div className="flex flex-wrap items-center gap-3 text-sm text-gray-600 font-medium">
                          <div className="flex items-center gap-1.5">
                            <Globe className="w-4 h-4 flex-shrink-0" />
                            <span className="truncate">{ref.domain}</span>
                          </div>
                          <span className="text-gray-400">•</span>
                          <div className="flex items-center gap-1.5">
                            <Shield className="w-4 h-4 flex-shrink-0" />
                            <span>{ref.credibility.label}</span>
                          </div>
                          <span className="text-gray-400">•</span>
                          <div className="flex items-center gap-1.5">
                            <Info className="w-4 h-4 flex-shrink-0" />
                            <span>{ref.similarity_percentage}% tương đồng</span>
                          </div>
                        </div>
                      </div>
                      
                      <ExternalLink className="w-6 h-6 text-gray-400 group-hover:text-blue-600 transition-colors flex-shrink-0" />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

      </div>
    </div>
  );
};

export default CheckerResult;