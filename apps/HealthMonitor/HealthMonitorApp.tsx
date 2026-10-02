import React, { useContext, useCallback } from 'react';
import { MemoryRouter, Routes, Route, useLocation, UNSAFE_NavigationContext } from 'react-router-dom';
import { useAppNavigationHandler } from '@/os/hooks/useAppNavigationHandler';
import { manifest } from './manifest';
import { useAppNavigate } from './navigation';
import { HomePage } from './pages/HomePage';
import { HealthDetailPage } from './pages/HealthDetailPage';

function NavigationHandler() {
  const location = useLocation();
  const { back } = useAppNavigate();
  const { navigator } = useContext(UNSAFE_NavigationContext);

  const onBack = useCallback((): boolean => {
    if (location.pathname !== '/') {
      back();
      return true;
    }
    return false;
  }, [back, location.pathname]);

  useAppNavigationHandler(manifest.id, { onBack });
  return null;
}

function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/details" element={<HealthDetailPage />} />
    </Routes>
  );
}

export default function HealthMonitorApp() {
  return (
    <MemoryRouter initialEntries={['/']}>
      <NavigationHandler />
      <AppRoutes />
    </MemoryRouter>
  );
}