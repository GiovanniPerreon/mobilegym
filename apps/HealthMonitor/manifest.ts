import type { AppManifest } from '@/os/types/manifest';
import { HealthIcon } from './res/icons';

export const manifest: AppManifest = {
  id: 'healthmonitor',
  packageName: 'com.example.healthmonitor',
  displayName: 'Health Monitor',
  displayNameEn: 'Health Monitor',
  version: '1.0.0',
  versionCode: 1,
  type: 'plugin',
  icon: HealthIcon,
  iconBackground: '#10b981',
  theme: {
    colors: {
      primary: '#10b981',
      background: '#f6f7f9',
    },
  },
};