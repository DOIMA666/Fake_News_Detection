import React from 'react';
import { Activity, CheckCircle, AlertCircle, Target, Clock, Flame } from 'lucide-react';
import StatCard from './StatCard';
import { getVerdictConfig } from '../utils/newsHelpers';

const Dashboard = ({ 
  dataLoading, 
  stats, 
  recentChecks, 
  trendingTopics, 
  setActiveTab 
}) => {
  return (
    <div className="space-y-8 animate-fade-in">
      {dataLoading && (
        <div className="text-center py-8">
          <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent mx-auto mb-4"></div>
          <p className="text-gray-600">Đang tải dữ liệu...</p>
        </div>
      )}

      {!dataLoading && stats.totalChecks === 0 && (
        <div className="bg-white rounded-2xl shadow-xl p-12 text-center border border-gray-100">
          <Activity className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-2xl font-bold text-gray-800 mb-2">Chưa có dữ liệu</h3>
          <p className="text-gray-600 mb-6">Hãy thử kiểm tra tin tức đầu tiên!</p>
          <button
            onClick={() => setActiveTab('checker')}
            className="px-6 py-3 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-xl font-semibold hover:shadow-lg transition-all"
          >
            Bắt đầu kiểm tra
          </button>
        </div>
      )}

      {!dataLoading && stats.totalChecks > 0 && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <StatCard
              icon={Activity}
              label="Tổng kiểm tra"
              value={stats.totalChecks.toLocaleString()}
              trend="up"
              color="text-blue-600"
              bgGradient="from-blue-500 via-blue-600 to-indigo-600"
            />
            <StatCard
              icon={CheckCircle}
              label="Tin đúng"
              value={stats.trueNews}
              trend="up"
              color="text-green-600"
              bgGradient="from-emerald-500 via-green-500 to-teal-600"
            />
            <StatCard
              icon={AlertCircle}
              label="Tin sai"
              value={stats.falseNews}
              trend="down"
              color="text-red-600"
              bgGradient="from-red-500 via-rose-500 to-pink-600"
            />
            <StatCard
              icon={Target}
              label="Độ chính xác"
              value={`${stats.accuracy}%`}
              trend="up"
              color="text-purple-600"
              bgGradient="from-purple-500 via-violet-500 to-purple-600"
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-bold text-gray-800 flex items-center gap-2">
                  <Clock className="w-6 h-6 text-blue-600" />
                  Hoạt động gần đây
                </h3>
              </div>
              <div className="space-y-4">
                {recentChecks.map((check, idx) => {
                  const config = getVerdictConfig(check.verdict);
                  return (
                    <div 
                      key={check.id} 
                      className="flex items-center gap-4 p-4 bg-gradient-to-r from-gray-50 to-blue-50/30 rounded-xl hover:shadow-md transition-all duration-300 group cursor-pointer"
                      style={{ animationDelay: `${idx * 100}ms` }}
                    >
                      <div className={`w-3 h-3 rounded-full bg-gradient-to-br ${config.bg} shadow-lg`}></div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-semibold text-gray-800 truncate group-hover:text-blue-600 transition-colors">
                          {check.title}
                        </p>
                        <p className="text-xs text-gray-500">{check.time}</p>
                      </div>
                      <div className="text-right">
                        <div className={`text-sm font-bold ${config.color}`}>
                          {check.confidence}%
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-bold text-gray-800 flex items-center gap-2">
                  <Flame className="w-6 h-6 text-orange-600" />
                  Chủ đề nổi bật
                </h3>
              </div>
              <div className="space-y-3">
                {trendingTopics.map((topic, idx) => (
                  <div 
                    key={idx} 
                    className="flex items-center justify-between p-4 bg-gradient-to-r from-orange-50 to-red-50/30 rounded-xl hover:shadow-md transition-all duration-300 group cursor-pointer"
                    style={{ animationDelay: `${idx * 100}ms` }}
                  >
                    <div className="flex items-center gap-3 flex-1">
                      <span className="text-2xl font-bold text-gray-300">#{idx + 1}</span>
                      <div>
                        <p className="text-sm font-bold text-gray-800 group-hover:text-orange-600 transition-colors">
                          {topic.topic}
                        </p>
                        <p className="text-xs text-gray-500">{topic.count} lượt kiểm tra</p>
                      </div>
                    </div>
                    <div className={`px-3 py-1 rounded-full text-xs font-bold ${
                      topic.trend === 'up' ? 'bg-green-100 text-green-700' :
                      topic.trend === 'down' ? 'bg-red-100 text-red-700' :
                      'bg-gray-100 text-gray-700'
                    }`}>
                      {topic.trend === 'up' ? '↑' : topic.trend === 'down' ? '↓' : '→'}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-2xl shadow-2xl p-8">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-white">
              <div className="text-center">
                <div className="text-4xl font-extrabold mb-2">{stats.todayChecks}</div>
                <div className="text-sm font-medium opacity-90">Hôm nay</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-extrabold mb-2">{Math.round(stats.accuracy)}%</div>
                <div className="text-sm font-medium opacity-90">Độ chính xác</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-extrabold mb-2">15+</div>
                <div className="text-sm font-medium opacity-90">Nguồn tin</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-extrabold mb-2">&lt;3s</div>
                <div className="text-sm font-medium opacity-90">Thời gian</div>
              </div>
            </div> 
          </div> 
        </>
      )}
    </div>
  );
};

export default Dashboard;