import React from 'react';
import { FileText, Globe, Search, Zap, AlertCircle } from 'lucide-react';
import CheckerResult from './CheckerResult';

const Checker = ({ 
  inputType, 
  setInputType, 
  content, 
  setContent, 
  loading, 
  progress, 
  handleSubmit, 
  handleKeyPress, 
  error, 
  result, 
  showResult, 
  saveResult, 
  shareResult, 
  exportResult,
  apiURL // ✅ Nhận prop mới
}) => {
  return (
    <div className="space-y-6 md:space-y-8">
      <div className="bg-white rounded-3xl shadow-xl p-6 md:p-10 border border-gray-100">
        
        {/* Nút chọn loại Input - Tối ưu Mobile */}
        <div className="flex flex-col sm:flex-row gap-3 md:gap-4 mb-6 md:mb-10">
          <button
            onClick={() => setInputType('text')}
            className={`flex-1 py-3 md:py-5 px-4 md:px-8 rounded-xl md:rounded-2xl font-semibold text-sm md:text-base transition-all duration-300 flex items-center justify-center ${
              inputType === 'text'
                ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white shadow-lg shadow-blue-500/50'
                : 'bg-white text-gray-700 hover:bg-gray-50 border-2 border-gray-200'
            }`}
          >
            <FileText className="w-5 h-5 mr-2" />
            Nhập văn bản
          </button>
          <button
            onClick={() => setInputType('url')}
            className={`flex-1 py-3 md:py-5 px-4 md:px-8 rounded-xl md:rounded-2xl font-semibold text-sm md:text-base transition-all duration-300 flex items-center justify-center ${
              inputType === 'url'
                ? 'bg-gradient-to-r from-purple-500 to-pink-600 text-white shadow-lg shadow-purple-500/50'
                : 'bg-white text-gray-700 hover:bg-gray-50 border-2 border-gray-200'
            }`}
          >
            <Globe className="w-5 h-5 mr-2" />
            Nhập URL
          </button>
        </div>

        <div className="mb-6 md:mb-8">
          <label className="block text-base md:text-lg font-bold text-gray-800 mb-3 flex items-center gap-2">
            {inputType === 'text' ? (
              <>
                <span className="text-xl md:text-2xl">📄</span>
                <span>Nội dung cần kiểm tra</span>
              </>
            ) : (
              <>
                <span className="text-xl md:text-2xl">🔗</span>
                <span>URL bài báo</span>
              </>
            )}
          </label>
          {inputType === 'text' ? (
            <textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="Dán nội dung tin tức cần kiểm tra vào đây..."
              className="w-full h-40 md:h-52 p-4 md:p-5 border-2 border-gray-200 rounded-xl md:rounded-2xl focus:ring-4 focus:ring-blue-500/30 focus:border-blue-500 transition-all text-sm md:text-base text-gray-800 resize-none"
              disabled={loading}
            />
          ) : (
            <input
              type="url"
              value={content}
              onChange={(e) => setContent(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="https://vnexpress.net/..."
              className="w-full p-4 md:p-5 border-2 border-gray-200 rounded-xl md:rounded-2xl focus:ring-4 focus:ring-purple-500/30 focus:border-purple-500 transition-all text-sm md:text-base text-gray-800"
              disabled={loading}
            />
          )}
        </div>

        <button
          onClick={handleSubmit}
          disabled={loading || !content.trim()}
          className="w-full bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white py-4 md:py-6 px-6 rounded-xl md:rounded-2xl font-bold text-base md:text-lg hover:shadow-xl active:scale-[0.98] transition-all duration-300 flex items-center justify-center gap-3 md:gap-4 disabled:opacity-70 disabled:cursor-not-allowed group relative overflow-hidden"
        >
          {loading ? (
            <>
              <div className="animate-spin rounded-full h-5 w-5 md:h-6 md:w-6 border-3 border-white border-t-transparent"></div>
              <span>Đang phân tích...</span>
            </>
          ) : (
            <>
              <Search className="w-5 h-5 md:w-6 md:h-6" />
              <span>Kiểm tra ngay</span>
              <Zap className="w-4 h-4 md:w-5 md:h-5" />
            </>
          )}
        </button>

        {loading && (
          <div className="mt-6 md:mt-8 animate-fade-in">
            <div className="h-2 md:h-3 bg-gray-100 rounded-full overflow-hidden shadow-inner">
              <div 
                className="h-full bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-600 transition-all duration-500 ease-out relative"
                style={{ width: `${progress}%` }}
              >
                <div className="absolute inset-0 bg-white/30 animate-pulse"></div>
              </div>
            </div>
            <p className="text-center text-xs md:text-sm text-gray-600 mt-2 font-medium">
              {progress < 30 ? 'Đang phân tích nội dung...' : 
               progress < 60 ? 'Đang tìm kiếm nguồn tin...' : 
               'Đang tổng hợp kết quả...'}
            </p>
          </div>
        )}

        {error && (
          <div className="mt-6 md:mt-8 p-4 md:p-6 bg-red-50 border-l-4 border-red-400 rounded-xl flex items-start gap-3 md:gap-4 animate-fade-in">
            <AlertCircle className="w-5 h-5 md:w-6 md:h-6 text-red-600 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-bold text-red-900 text-base md:text-lg">Đã xảy ra lỗi</p>
              <p className="text-red-700 text-xs md:text-sm whitespace-pre-line">{error}</p>
            </div>
          </div>
        )}
      </div>

      <CheckerResult 
        result={result}
        showResult={showResult}
        saveResult={saveResult}
        shareResult={shareResult}
        exportResult={exportResult}
        apiURL={apiURL} // ✅ Truyền xuống để gọi feedback
      />
    </div>
  );
};

export default Checker;