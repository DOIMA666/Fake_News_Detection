import React, { useState, useEffect } from 'react';
import { AlertCircle, CheckCircle, Search, ExternalLink, Globe, FileText, Sparkles, Shield, TrendingUp, Info } from 'lucide-react';

const FakeNewsChecker = () => {
  const [inputType, setInputType] = useState('text');
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);
  const [showResult, setShowResult] = useState(false);

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
        throw new Error(data.detail || 'Có lỗi xảy ra');
      }

      if (data.success) {
        setProgress(100);
        setTimeout(() => setResult(data), 300);
      } else {
        setError(data.message || 'Không thể xử lý yêu cầu');
      }

    } catch (err) {
      setError(err.message || 'Lỗi kết nối đến server');
    } finally {
      setLoading(false);
    }
  };

  const getVerdictConfig = (verdict) => {
    const configs = {
      'HIGHLY_LIKELY_TRUE': {
        bg: 'from-green-400 to-emerald-500',
        border: 'border-green-400',
        text: 'text-green-800',
        icon: CheckCircle,
        glow: 'shadow-green-500/50'
      },
      'LIKELY_TRUE': {
        bg: 'from-green-300 to-green-400',
        border: 'border-green-300',
        text: 'text-green-700',
        icon: CheckCircle,
        glow: 'shadow-green-400/50'
      },
      'UNCERTAIN': {
        bg: 'from-yellow-400 to-orange-400',
        border: 'border-orange-300',
        text: 'text-orange-800',
        icon: Info,
        glow: 'shadow-orange-400/50'
      },
      'LIKELY_FALSE': {
        bg: 'from-red-400 to-red-500',
        border: 'border-red-300',
        text: 'text-red-700',
        icon: AlertCircle,
        glow: 'shadow-red-400/50'
      },
      'HIGHLY_LIKELY_FALSE': {
        bg: 'from-red-500 to-red-600',
        border: 'border-red-400',
        text: 'text-red-800',
        icon: AlertCircle,
        glow: 'shadow-red-500/50'
      }
    };
    return configs[verdict] || configs['UNCERTAIN'];
  };

  const CircularProgress = ({ value, label }) => {
    const radius = 40;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (value / 100) * circumference;

    return (
      <div className="flex flex-col items-center justify-center">
        
        {/* 2. Giữ container này để chứa vòng tròn và số % */}
        <div className="relative inline-flex items-center justify-center">
          <svg className="transform -rotate-90 w-32 h-32">
            <circle
              cx="64"
              cy="64"
              r={radius}
              stroke="currentColor"
              strokeWidth="8"
              fill="none"
              className="text-gray-200"
            />
            <circle
              cx="64"
              cy="64"
              r={radius}
              stroke="currentColor"
              strokeWidth="8"
              fill="none"
              strokeDasharray={circumference}
              strokeDashoffset={offset}
              className="text-blue-600 transition-all duration-1000 ease-out"
              strokeLinecap="round"
            />
          </svg>
          {/* 3. Div "absolute" này BÂY GIỜ CHỈ chứa số % */}
          <div className="absolute flex flex-col items-center">
            <span className="text-3xl font-bold text-gray-800">{Math.round(value)}%</span>
            {/* Đã xóa label khỏi đây */}
          </div>
        </div>
        
        {/* 4. Thêm label (chữ) BÊN NGOÀI, nằm bên dưới vòng tròn */}
        <span className="text-sm font-semibold text-gray-700 mt-2">{label}</span>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900 py-8 px-4 relative overflow-hidden">
      {/* Animated background */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-2000"></div>
        <div className="absolute top-40 left-1/2 w-80 h-80 bg-pink-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-4000"></div>
      </div>

      <style>{`
        @keyframes blob {
          0%, 100% { transform: translate(0, 0) scale(1); }
          33% { transform: translate(30px, -50px) scale(1.1); }
          66% { transform: translate(-20px, 20px) scale(0.9); }
        }
        .animate-blob {
          animation: blob 7s infinite;
        }
        .animation-delay-2000 {
          animation-delay: 2s;
        }
        .animation-delay-4000 {
          animation-delay: 4s;
        }
        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-10px); }
        }
        .animate-float {
          animation: float 3s ease-in-out infinite;
        }
      `}</style>

      <div className="max-w-5xl mx-auto relative z-10">
        {/* Header */}
        <div className="text-center mb-12 animate-float">
          <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl shadow-2xl mb-6 transform hover:scale-110 transition-transform duration-300">
            <Shield className="w-10 h-10 text-white" />
          </div>
          <h1 className="text-5xl md:text-6xl font-bold text-white mb-3 tracking-tight">
            Kiểm Tra Tin Giả
          </h1>
          <p className="text-xl text-blue-200 flex items-center justify-center gap-2">
            <Sparkles className="w-5 h-5" />
            Phát hiện tin giả bằng AI thông minh
          </p>
        </div>

        {/* Main Card */}
        <div className="bg-white/95 backdrop-blur-xl rounded-3xl shadow-2xl p-8 mb-8 border border-white/20 transform transition-all duration-500 hover:shadow-blue-500/20">
          {/* Type Selector */}
          <div className="flex gap-4 mb-8">
            <button
              onClick={() => setInputType('text')}
              className={`flex-1 py-4 px-6 rounded-2xl font-semibold transition-all duration-300 transform hover:scale-105 ${
                inputType === 'text'
                  ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-xl shadow-blue-500/50'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <FileText className="inline-block w-6 h-6 mr-2" />
              Nhập văn bản
            </button>
            <button
              onClick={() => setInputType('url')}
              className={`flex-1 py-4 px-6 rounded-2xl font-semibold transition-all duration-300 transform hover:scale-105 ${
                inputType === 'url'
                  ? 'bg-gradient-to-r from-purple-600 to-purple-700 text-white shadow-xl shadow-purple-500/50'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <Globe className="inline-block w-6 h-6 mr-2" />
              Nhập URL
            </button>
          </div>

          {/* Input Area */}
          <div className="mb-6">
            <label className="block text-lg font-semibold text-gray-800 mb-3">
              {inputType === 'text' ? '📝 Nội dung cần kiểm tra:' : '🔗 URL bài báo:'}
            </label>
            {inputType === 'text' ? (
              <textarea
                value={content}
                onChange={(e) => setContent(e.target.value)}
                placeholder="Dán nội dung tin tức cần kiểm tra vào đây..."
                className="w-full h-48 p-4 border-2 border-gray-200 rounded-2xl focus:ring-4 focus:ring-blue-500/50 focus:border-blue-500 transition-all duration-300 resize-none text-gray-800"
                disabled={loading}
              />
            ) : (
              <>
                <input
                  type="url"
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  placeholder="https://vnexpress.net/tieu-de-bai-bao-12345.html"
                  className="w-full p-4 border-2 border-gray-200 rounded-2xl focus:ring-4 focus:ring-purple-500/50 focus:border-purple-500 transition-all duration-300 text-gray-800"
                  disabled={loading}
                />
                <div className="mt-4 p-4 bg-gradient-to-r from-blue-50 to-purple-50 border-2 border-blue-200 rounded-2xl">
                  <p className="text-sm text-blue-900 font-medium">
                    <strong>💡 Lưu ý:</strong> Nhập URL của <strong>bài báo cụ thể</strong>
                  </p>
                  <p className="text-xs text-blue-700 mt-2 font-mono">
                    ✅ Đúng: https://vnexpress.net/tieu-de-bai-viet-12345.html<br/>
                    ❌ Sai: https://vnexpress.net/topic/covid-19
                  </p>
                </div>
              </>
            )}
          </div>

          {/* Submit Button */}
          <button
            onClick={handleSubmit}
            disabled={loading || !content.trim()}
            className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-5 px-8 rounded-2xl font-bold text-lg hover:from-blue-700 hover:to-purple-700 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed transition-all duration-300 flex items-center justify-center gap-3 shadow-xl hover:shadow-2xl transform hover:scale-105 active:scale-95"
          >
            {loading ? (
              <>
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></div>
                Đang phân tích...
              </>
            ) : (
              <>
                <Search className="w-6 h-6" />
                Kiểm tra ngay
              </>
            )}
          </button>

          {/* Progress Bar */}
          {loading && (
            <div className="mt-6">
              <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-blue-600 to-purple-600 transition-all duration-500 ease-out"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="text-center text-sm text-gray-600 mt-2">
                {progress < 30 ? 'Đang phân tích nội dung...' : 
                 progress < 60 ? 'Đang tìm kiếm nguồn tin...' : 
                 progress < 90 ? 'Đang so sánh với các nguồn uy tín...' : 
                 'Gần hoàn thành...'}
              </p>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="mt-6 p-5 bg-gradient-to-r from-red-50 to-pink-50 border-2 border-red-200 rounded-2xl flex items-start gap-4 animate-pulse">
              <AlertCircle className="w-6 h-6 text-red-600 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-bold text-red-900 text-lg">Lỗi</p>
                <p className="text-red-700">{error}</p>
              </div>
            </div>
          )}
        </div>

        {/* Result Card */}
        {result && result.verdict && (
          <div className={`transition-all duration-700 transform ${showResult ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
            <div className="bg-white/95 backdrop-blur-xl rounded-3xl shadow-2xl p-8 border border-white/20">
              {/* Verdict Header */}
              <div className={`relative p-8 rounded-3xl mb-8 bg-gradient-to-br ${getVerdictConfig(result.verdict.code).bg} shadow-2xl ${getVerdictConfig(result.verdict.code).glow} overflow-hidden`}>
                <div className="absolute inset-0 bg-white/10 backdrop-blur-sm"></div>
                <div className="relative z-10">
                  <div className="flex items-center justify-between mb-6">
                    <h2 className="text-3xl font-bold text-white drop-shadow-lg">Kết quả kiểm tra</h2>
                    {React.createElement(getVerdictConfig(result.verdict.code).icon, {
                      className: "w-12 h-12 text-white drop-shadow-lg animate-pulse"
                    })}
                  </div>
                  
                  <p className="text-2xl font-bold text-white mb-3 drop-shadow-md">{result.verdict.label}</p>
                  <p className="text-white/90 text-lg drop-shadow-sm">{result.verdict.explanation}</p>
                  
                  {/* Stats */}
                  <div className="grid grid-cols-2 gap-6 mt-8">
                    <div className="bg-white/20 backdrop-blur-sm p-5 rounded-2xl border border-white/30 transform hover:scale-105 transition-transform duration-300">
                      <CircularProgress 
                        value={result.verdict.similarity_percentage} 
                        label="Tương đồng"
                      />
                    </div>
                    <div className="bg-white/20 backdrop-blur-sm p-5 rounded-2xl border border-white/30 transform hover:scale-105 transition-transform duration-300">
                      <CircularProgress 
                        value={result.verdict.confidence_percentage} 
                        label="Tin cậy"
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Keywords */}
              {result.keywords && result.keywords.length > 0 && (
                <div className="mb-8 animate-fade-in">
                  <h3 className="font-bold text-gray-900 mb-4 text-xl flex items-center gap-2">
                    <TrendingUp className="w-6 h-6 text-blue-600" />
                    Từ khóa chính
                  </h3>
                  <div className="flex flex-wrap gap-3">
                    {result.keywords.map((keyword, idx) => (
                      <span 
                        key={idx} 
                        className="px-4 py-2 bg-gradient-to-r from-blue-100 to-purple-100 text-blue-800 rounded-full text-sm font-medium border-2 border-blue-200 transform hover:scale-110 transition-all duration-300 cursor-default shadow-md hover:shadow-lg"
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
                  <h3 className="font-bold text-gray-900 mb-5 text-xl flex items-center gap-2">
                    <Shield className="w-6 h-6 text-green-600" />
                    Nguồn tin uy tín
                  </h3>
                  <div className="space-y-4">
                    {result.references.map((ref, idx) => (
                      <div 
                        key={idx} 
                        className="group p-6 border-2 border-gray-200 rounded-2xl hover:border-blue-400 hover:shadow-xl transition-all duration-300 bg-gradient-to-r from-white to-gray-50 transform hover:-translate-y-1"
                        style={{
                          animationDelay: `${idx * 150}ms`
                        }}
                      >
                        <div className="flex items-start justify-between gap-4">
                          <div className="flex-1">
                            <a 
                              href={ref.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-blue-600 hover:text-blue-800 font-semibold mb-2 block text-lg group-hover:underline"
                            >
                              {ref.title || ref.url}
                            </a>
                            <p className="text-sm text-gray-600 font-medium">{ref.domain}</p>
                          </div>
                          <div className="flex items-center gap-3 flex-shrink-0">
                            <div className="text-center">
                              <div className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                                {ref.similarity_percentage}%
                              </div>
                              <div className="text-xs text-gray-500">tương đồng</div>
                            </div>
                            <ExternalLink className="w-5 h-5 text-gray-400 group-hover:text-blue-600 transition-colors" />
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
        <div className="text-center mt-12 text-blue-200">
          <p className="text-lg font-medium">🤖 Hệ thống AI phân tích thông minh</p>
          <p className="mt-2 text-sm text-blue-300">Kết quả mang tính chất tham khảo</p>
        </div>
      </div>
    </div>
  );
};

export default FakeNewsChecker;