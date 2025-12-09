import React from 'react';
import { Users, Globe, XCircle, Info, Clock, Activity, Eye, Bookmark } from 'lucide-react';
import { getVerdictConfig } from '../utils/newsHelpers';

const History = ({ 
  historySubTab, 
  setHistorySubTab, 
  history, 
  setHistory, 
  communityHistory, 
  loadCommunityHistory, 
  savedItems, 
  deleteSavedItem ,
  onSelect
}) => {
  return (
    <div className="space-y-6 animate-fade-in">
      {/* Sub-tabs cho History */}
      <div className="flex justify-center mb-6">
        <div className="inline-flex bg-white rounded-2xl shadow-lg p-2 border border-gray-200">
          <button
            onClick={() => setHistorySubTab('personal')}
            className={`flex items-center gap-2 px-6 py-3 rounded-xl font-semibold transition-all duration-300 ${
              historySubTab === 'personal'
                ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white shadow-lg'
                : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
            }`}
          >
            <Users className="w-5 h-5" />
            Của tôi
            <span className="px-2 py-1 bg-white/20 rounded-full text-xs">
              {history.length}
            </span>
          </button>
          <button
            onClick={() => setHistorySubTab('community')}
            className={`flex items-center gap-2 px-6 py-3 rounded-xl font-semibold transition-all duration-300 ${
              historySubTab === 'community'
                ? 'bg-gradient-to-r from-purple-500 to-pink-600 text-white shadow-lg'
                : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
            }`}
          >
            <Globe className="w-5 h-5" />
            Cộng đồng
          </button>
        </div>
      </div>

      {/* Personal History (LocalStorage) */}
      {historySubTab === 'personal' && (
        <div className="bg-white rounded-3xl shadow-xl p-8 border border-gray-100">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-3">
              <Users className="w-8 h-8 text-blue-600" />
              Lịch sử của tôi
              <span className="text-sm font-normal text-gray-500">({history.length} mục)</span>
            </h2>
            {history.length > 0 && (
              <button 
                onClick={() => {
                  if (confirm('Xóa lịch sử cá nhân? (Không ảnh hưởng dữ liệu cộng đồng)')) {
                    setHistory([]);
                    localStorage.removeItem('checkHistory');
                    alert('Đã xóa lịch sử cá nhân!');
                  }
                }}
                className="px-4 py-2 bg-red-50 text-red-600 rounded-xl hover:bg-red-100 transition-colors font-semibold text-sm flex items-center gap-2"
              >
                <XCircle className="w-4 h-4" />
                Xóa tất cả
              </button>
            )}
          </div>

          <div className="mb-4 p-4 bg-blue-50 border-l-4 border-blue-400 rounded-lg">
            <p className="text-sm text-blue-800 flex items-center gap-2">
              <Info className="w-4 h-4" />
              <strong>Chỉ bạn thấy:</strong> Dữ liệu lưu trên trình duyệt của bạn (LocalStorage)
            </p>
          </div>

          {history.length === 0 ? (
            <div className="text-center py-16">
              <Clock className="w-16 h-16 text-gray-300 mx-auto mb-4" />
              <p className="text-gray-500 text-lg">Chưa có lịch sử kiểm tra cá nhân</p>
              <p className="text-gray-400 text-sm mt-2">Các kết quả kiểm tra của bạn sẽ được lưu tại đây</p>
            </div>
          ) : (
            <div className="space-y-4">
              {history.map((item, idx) => {
                const config = getVerdictConfig(item.verdict.code);
                const VerdictIcon = config.icon;
                return (
                  <div 
                    key={item.id}
                    onClick={() => onSelect(item, false)}
                    className={`p-6 rounded-2xl border-2 ${config.bgLight} ${config.bg.replace('from-', 'border-').split(' ')[0].replace('via-', '').replace('to-', '')} hover:shadow-lg transition-all duration-300 cursor-pointer`}
                    style={{ animationDelay: `${idx * 50}ms` }}
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <VerdictIcon className={`w-6 h-6 ${config.color}`} />
                          <span className={`font-bold ${config.color}`}>{item.verdict.label}</span>
                          <span className="px-3 py-1 bg-white rounded-full text-xs font-bold text-gray-600">
                            {Math.round(item.verdict.confidence_percentage)}%
                          </span>
                        </div>
                        <p className="text-gray-700 text-sm leading-relaxed mb-2">{item.content}</p>
                        <div className="flex items-center gap-3 text-xs text-gray-500">
                          <span>{item.type === 'url' ? '🔗 URL' : '📄 Text'}</span>
                          <span>•</span>
                          <span>{new Date(item.timestamp).toLocaleString('vi-VN')}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Community Feed (Backend Database) */}
      {historySubTab === 'community' && (
        <div className="bg-white rounded-3xl shadow-xl p-8 border border-gray-100">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-3">
              <Globe className="w-8 h-8 text-purple-600" />
              Hoạt động gần đây của cộng đồng
              <span className="text-sm font-normal text-gray-500">({communityHistory.length} mục)</span>
            </h2>
            <button 
              onClick={loadCommunityHistory}
              className="px-4 py-2 bg-purple-50 text-purple-600 rounded-xl hover:bg-purple-100 transition-colors font-semibold text-sm flex items-center gap-2"
            >
              <Activity className="w-4 h-4" />
              Làm mới
            </button>
          </div>

          <div className="mb-4 p-4 bg-purple-50 border-l-4 border-purple-400 rounded-lg">
            <p className="text-sm text-purple-800 flex items-center gap-2">
              <Eye className="w-4 h-4" />
              <strong>Công khai:</strong> Xem tin tức mà mọi người đang kiểm tra (20 tin gần nhất)
            </p>
          </div>

          {communityHistory.length === 0 ? (
            <div className="text-center py-16">
              <Globe className="w-16 h-16 text-gray-300 mx-auto mb-4" />
              <p className="text-gray-500 text-lg">Chưa có hoạt động nào</p>
              <p className="text-gray-400 text-sm mt-2">Hãy là người đầu tiên kiểm tra tin tức!</p>
            </div>
          ) : (
            <div className="space-y-4">
              {communityHistory.map((item, idx) => {
                const verdictCode = item.verdict_code || 'UNCERTAIN';
                const config = getVerdictConfig(verdictCode);
                const VerdictIcon = config.icon;
                
                return (
                  <div 
                    key={item.id}
                    onClick={() => onSelect(item, true)}
                    className={`p-6 rounded-2xl border-2 ${config.bgLight} hover:shadow-lg transition-all duration-300`}
                    style={{ animationDelay: `${idx * 50}ms` }}
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <VerdictIcon className={`w-6 h-6 ${config.color}`} />
                          <span className={`font-bold ${config.color}`}>
                            {item.verdict_label || config.badge}
                          </span>
                          <span className="px-3 py-1 bg-white rounded-full text-xs font-bold text-gray-600">
                            {Math.round(item.confidence_percentage || 0)}%
                          </span>
                          <span className="ml-auto text-xs text-gray-400">
                            {new Date(item.timestamp).toLocaleString('vi-VN')}
                          </span>
                        </div>
                        <p className="text-gray-700 text-sm leading-relaxed mb-2">
                          {item.content_preview}
                        </p>
                        <div className="flex items-center gap-3 text-xs text-gray-500">
                          <span>🔍 {item.num_references || 15} nguồn</span>
                          <span>•</span>
                          <span>{item.input_type === 'url' ? '🔗 URL' : '📄 Text'}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Saved Items */}
      {savedItems.length > 0 && (
        <div className="bg-white rounded-3xl shadow-xl p-8 border border-gray-100">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-3">
              <Bookmark className="w-8 h-8 text-amber-600" />
              Đã lưu
              <span className="text-sm font-normal text-gray-500">({savedItems.length} mục)</span>
            </h2>
          </div>

          <div className="space-y-4">
            {savedItems.map((item, idx) => {
              const config = getVerdictConfig(item.result.verdict.code);
              return (
                <div 
                  key={item.id}
                  onClick={() => onSelect(item, false)}
                  className={`p-6 rounded-2xl border-2 ${config.bgLight} hover:shadow-lg transition-all duration-300`}
                  style={{ animationDelay: `${idx * 50}ms` }}
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <span className={`font-bold ${config.color}`}>{item.result.verdict.label}</span>
                        <span className="px-3 py-1 bg-white rounded-full text-xs font-bold text-gray-600">
                          {Math.round(item.result.verdict.confidence_percentage)}%
                        </span>
                      </div>
                      <p className="text-gray-700 text-sm">{item.content}</p>
                      <p className="text-xs text-gray-500 mt-2">
                        {new Date(item.timestamp).toLocaleString('vi-VN')}
                      </p>
                    </div>
                    <button
                      onClick={() => deleteSavedItem(item.id)}
                      className="p-2 hover:bg-red-100 rounded-lg transition-colors text-red-600"
                      title="Xóa"
                    >
                      <XCircle className="w-5 h-5" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

export default History;