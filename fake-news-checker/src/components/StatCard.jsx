import React from 'react';

const StatCard = ({ icon, label, value, trend, color, bgGradient }) => {
  const IconComp = icon;
  return (
    <div className={`relative overflow-hidden bg-gradient-to-br ${bgGradient} rounded-2xl p-6 shadow-lg hover:shadow-2xl transition-all duration-500 transform hover:scale-105 group`}>
      <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full -mr-16 -mt-16 group-hover:scale-150 transition-transform duration-700"></div>
      <div className="relative z-10">
        <div className="flex items-center justify-between mb-4">
          <div className={`w-14 h-14 ${color} bg-white/20 backdrop-blur-sm rounded-xl flex items-center justify-center shadow-lg`}>
            <IconComp className="w-7 h-7" />
          </div>
          {trend && (
            <span className={`text-xs font-bold px-3 py-1 rounded-full ${trend === 'up' ? 'bg-green-500/20 text-green-100' : 'bg-red-500/20 text-red-100'}`}>
              {trend === 'up' ? '↑' : '↓'} {Math.abs(Math.random() * 20).toFixed(1)}%
            </span>
          )}
        </div>
        <div className="text-3xl font-extrabold text-white mb-1">{value}</div>
        <div className="text-sm text-white/80 font-medium">{label}</div>
      </div>
    </div>
  );
};

export default StatCard;