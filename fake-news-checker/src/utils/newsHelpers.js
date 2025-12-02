import { CheckCircle, Info, AlertCircle, XCircle } from 'lucide-react';

export const getVerdictConfig = (verdict) => {
  const configs = {
    // MỨC 1: ĐÚNG (Gộp màu xanh)
    'LIKELY_TRUE': {
      bg: 'from-emerald-500 via-green-500 to-teal-500',
      badge: 'Có khả năng đúng',
      color: 'text-emerald-600',
      bgLight: 'bg-emerald-50',
      icon: CheckCircle
    },
    
    // MỨC 2: SAI (Gộp màu đỏ)
    'LIKELY_FALSE': {
      bg: 'from-red-600 via-rose-600 to-pink-600',
      badge: 'Có khả năng sai',
      color: 'text-red-600',
      bgLight: 'bg-red-50',
      icon: XCircle
    },

    // MỨC 3: KHÔNG CHẮC (Màu vàng)
    'UNCERTAIN': {
      bg: 'from-amber-400 via-yellow-400 to-orange-400',
      badge: 'Không chắc chắn',
      color: 'text-amber-600',
      bgLight: 'bg-amber-50',
      icon: Info
    }
  };
  return configs[verdict] || configs['UNCERTAIN'];
};

export const getStanceConfig = (stance) => {
  const configs = {
    'SUPPORT': {
      label: 'Xác nhận',
      icon: CheckCircle,
      badge: 'bg-green-100 text-green-700',
      color: 'text-green-600'
    },
    'REFUTE': {
      label: 'Bác bỏ',
      icon: AlertCircle,
      badge: 'bg-red-100 text-red-700',
      color: 'text-red-600'
    },
    'DISCUSS': {
      label: 'Bàn luận',
      icon: Info,
      badge: 'bg-yellow-100 text-yellow-700',
      color: 'text-yellow-600'
    },
    'UNRELATED': {
      label: 'Không liên quan',
      icon: Info,
      badge: 'bg-gray-100 text-gray-700',
      color: 'text-gray-600'
    }
  };
  const stanceCode = typeof stance === 'string' ? stance : (stance?.code || 'DISCUSS');
  return configs[stanceCode.toUpperCase()] || configs['DISCUSS'];
};

export const getCategoryInfo = (catCode) => {
  const map = {
    'politics': { label: 'Chính trị', icon: '🏛️', color: 'bg-blue-50 text-blue-700 border-blue-200' },
    'crime': { label: 'Pháp luật', icon: '⚖️', color: 'bg-red-50 text-red-700 border-red-200' },
    'health': { label: 'Y tế', icon: '🏥', color: 'bg-green-50 text-green-700 border-green-200' },
    'entertainment': { label: 'Giải trí', icon: '🎬', color: 'bg-purple-50 text-purple-700 border-purple-200' },
    'sports': { label: 'Thể thao', icon: '⚽', color: 'bg-orange-50 text-orange-700 border-orange-200' },
    'economy': { label: 'Kinh tế', icon: '💰', color: 'bg-yellow-50 text-yellow-700 border-yellow-200' },
    'technology': { label: 'Công nghệ', icon: '💻', color: 'bg-indigo-50 text-indigo-700 border-indigo-200' },
    'other': { label: 'Tin tức', icon: '📰', color: 'bg-gray-50 text-gray-700 border-gray-200' }
  };
  return map[catCode] || map['other'];
};