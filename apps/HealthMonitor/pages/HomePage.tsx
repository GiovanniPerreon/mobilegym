import React, { useState, useEffect } from 'react';
import { useAppNavigate } from '../navigation';

// Types
interface HealthData {
  heartRate: number;
  systolic: number;
  diastolic: number;
  spo2: number;
  steps: number;
  calories: number;
  sleepHours: number;
  timestamp: Date;
}

// Fixed initial health data (score will be below 70)
const getInitialHealthData = (): HealthData => ({
  heartRate: 95,        // slightly high, contributes to lower score
  systolic: 135,
  diastolic: 85,
  spo2: 94,
  steps: 3000,
  calories: 200,
  sleepHours: 5.5,
  timestamp: new Date(),
});

// Generate random mock data (for refresh)
const generateRandomHealthData = (): HealthData => ({
  heartRate: Math.floor(Math.random() * (95 - 65 + 1) + 65),
  systolic: Math.floor(Math.random() * (135 - 105 + 1) + 105),
  diastolic: Math.floor(Math.random() * (90 - 70 + 1) + 70),
  spo2: Math.floor(Math.random() * (100 - 95 + 1) + 95),
  steps: Math.floor(Math.random() * 12000),
  calories: Math.floor(Math.random() * 800),
  sleepHours: +(Math.random() * 4 + 5).toFixed(1),
  timestamp: new Date(),
});

// Helper to compute health score (0-100)
const computeHealthScore = (data: HealthData): number => {
  let score = 0;
  if (data.heartRate >= 60 && data.heartRate <= 100) score += 25;
  else if (data.heartRate > 100) score += 15;
  else score += 20;
  if (data.systolic < 120 && data.diastolic < 80) score += 25;
  else if (data.systolic < 130 && data.diastolic < 85) score += 15;
  else score += 5;
  if (data.spo2 >= 95) score += 20;
  else if (data.spo2 >= 90) score += 10;
  if (data.steps > 8000) score += 15;
  else if (data.steps > 5000) score += 10;
  else score += 5;
  if (data.sleepHours >= 7) score += 15;
  else if (data.sleepHours >= 6) score += 8;
  return Math.min(100, score);
};

export const HomePage: React.FC = () => {
  const { go } = useAppNavigate();
  const [health, setHealth] = useState<HealthData>(getInitialHealthData());
  const [healthScore, setHealthScore] = useState(computeHealthScore(getInitialHealthData()));
  const [lastUpdated, setLastUpdated] = useState(new Date());

  const refreshData = () => {
    const newData = generateRandomHealthData();
    setHealth(newData);
    setHealthScore(computeHealthScore(newData));
    setLastUpdated(new Date());
  };

  // No automatic interval – only manual refresh
  useEffect(() => {
    // No auto‑refresh
  }, []);

  // Inject health score into simulator global state
  useEffect(() => {
    const setSimState = (score: number) => {
      if (window.__SIM__ && typeof window.__SIM__.setState === 'function') {
        window.__SIM__.setState(
          {
            apps: {
              healthmonitor: { healthScore: score },
            },
          },
          { deep: true }
        );
      } else {
        console.warn('__SIM__.setState not available yet');
      }
    };
    setSimState(healthScore);
  }, [healthScore]);

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getHrStatus = (hr: number) => {
    if (hr < 60) return { text: 'Low', color: 'text-blue-500' };
    if (hr <= 100) return { text: 'Normal', color: 'text-green-500' };
    return { text: 'Elevated', color: 'text-red-500' };
  };

  const hrStatus = getHrStatus(health.heartRate);

  return (
    <div className="p-4 pt-14 space-y-4 bg-gray-50 min-h-screen">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-green-600">Health Monitor</h1>
        <button onClick={refreshData} className="text-sm bg-green-100 text-green-700 px-3 py-1 rounded-full">
          ↻ Refresh
        </button>
      </div>

      <div className="bg-white rounded-xl shadow-sm p-4 text-center">
        <p className="text-gray-500 text-sm">Overall Health Score</p>
        <p className={`text-4xl font-bold ${getScoreColor(healthScore)}`}>{healthScore}</p>
        <p className="text-xs text-gray-400">Last updated: {lastUpdated.toLocaleTimeString()}</p>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="bg-white rounded-xl shadow-sm p-3">
          <div className="flex justify-between items-center">
            <span className="text-2xl">❤️</span>
            <span className={`text-xs font-medium ${hrStatus.color}`}>{hrStatus.text}</span>
          </div>
          <p className="text-gray-500 text-sm">Heart Rate</p>
          <p className="text-2xl font-semibold">{health.heartRate} <span className="text-sm">bpm</span></p>
        </div>
        <div className="bg-white rounded-xl shadow-sm p-3">
          <span className="text-2xl">🩸</span>
          <p className="text-gray-500 text-sm">Blood Pressure</p>
          <p className="text-2xl font-semibold">{health.systolic}/{health.diastolic}</p>
        </div>
        <div className="bg-white rounded-xl shadow-sm p-3">
          <span className="text-2xl">💨</span>
          <p className="text-gray-500 text-sm">SpO2</p>
          <p className="text-2xl font-semibold">{health.spo2}%</p>
        </div>
        <div className="bg-white rounded-xl shadow-sm p-3">
          <span className="text-2xl">😴</span>
          <p className="text-gray-500 text-sm">Sleep</p>
          <p className="text-2xl font-semibold">{health.sleepHours} <span className="text-sm">hrs</span></p>
        </div>
        <div className="bg-white rounded-xl shadow-sm p-3">
          <span className="text-2xl">👣</span>
          <p className="text-gray-500 text-sm">Steps</p>
          <p className="text-2xl font-semibold">{health.steps.toLocaleString()}</p>
        </div>
        <div className="bg-white rounded-xl shadow-sm p-3">
          <span className="text-2xl">🔥</span>
          <p className="text-gray-500 text-sm">Calories</p>
          <p className="text-2xl font-semibold">{health.calories}</p>
        </div>
      </div>

      <button onClick={() => go('open_details')} className="w-full bg-green-500 text-white py-3 rounded-xl font-medium shadow-md mt-2">
        View Detailed Report
      </button>
    </div>
  );
};