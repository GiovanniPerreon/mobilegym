import React from 'react';
import { useAppNavigate } from '../navigation';

export const HealthDetailPage: React.FC = () => {
  const { go } = useAppNavigate();

  const history = [
    { day: 'Mon', steps: 8200, sleep: 7.2, hr: 72 },
    { day: 'Tue', steps: 9500, sleep: 6.8, hr: 74 },
    { day: 'Wed', steps: 7800, sleep: 8.1, hr: 70 },
    { day: 'Thu', steps: 10200, sleep: 7.5, hr: 71 },
    { day: 'Fri', steps: 8900, sleep: 6.5, hr: 75 },
    { day: 'Sat', steps: 11500, sleep: 8.5, hr: 68 },
    { day: 'Sun', steps: 6300, sleep: 7.0, hr: 73 },
  ];

  const maxSteps = Math.max(...history.map(d => d.steps));

  return (
    <div className="p-4 pt-14 space-y-5 bg-gray-50 h-screen overflow-y-auto">
      {/* Back button and title */}
      <div className="flex items-center justify-between">
        <button onClick={() => go('go_back')} className="text-green-600 text-lg font-medium">
          ← Back
        </button>
        <h1 className="text-xl font-bold">Detailed Report</h1>
        <div className="w-6" />
      </div>

      {/* Summary Card */}
      <div className="bg-gradient-to-r from-green-500 to-teal-500 rounded-xl p-4 text-white">
        <p className="text-sm opacity-90">Weekly Summary</p>
        <p className="text-2xl font-bold mt-1">Great progress!</p>
        <p className="text-sm mt-2">You walked 62,400 steps this week. Keep it up!</p>
      </div>

      {/* Steps Chart */}
      <div className="bg-white rounded-xl p-4 shadow-sm">
        <h2 className="font-semibold mb-3">📊 Steps (last 7 days)</h2>
        <div className="flex justify-between items-end h-32 gap-1">
          {history.map((day, idx) => (
            <div key={idx} className="flex-1 flex flex-col items-center">
              <div
                className="w-full bg-green-200 rounded-t"
                style={{ height: `${(day.steps / maxSteps) * 80}px` }}
              />
              <span className="text-xs text-gray-500 mt-1">{day.day}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Sleep & Heart Rate Table */}
      <div className="bg-white rounded-xl p-4 shadow-sm">
        <h2 className="font-semibold mb-2">😴 Sleep & Heart Rate</h2>
        <div className="space-y-2">
          {history.map((day, idx) => (
            <div key={idx} className="flex justify-between text-sm border-b pb-1">
              <span className="text-gray-500">{day.day}</span>
              <span>😴 {day.sleep}h</span>
              <span>❤️ {day.hr} bpm</span>
            </div>
          ))}
        </div>
      </div>

      {/* Health Tip */}
      <div className="bg-blue-50 rounded-xl p-4 border border-blue-100">
        <p className="text-blue-800 text-sm">
          💡 Tip: Try to walk 10,000 steps daily and sleep 7-8 hours for optimal health.
        </p>
      </div>
    </div>
  );
};