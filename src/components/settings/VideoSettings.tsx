'use client';

import { Select, SelectOption } from '../ui/select';
import { useSettingsStore } from '../../stores/settingsStore';

export function VideoSettings() {
  const { preferences, updatePreference } = useSettingsStore();

  const videoQualityOptions: SelectOption[] = [
    { value: 'low', label: 'Low (720p)', description: 'Faster processing, smaller file size' },
    { value: 'medium', label: 'Medium (1080p)', description: 'Balanced quality and performance' },
    { value: 'high', label: 'High (1440p)', description: 'High quality, larger file size' },
    { value: 'ultra', label: 'Ultra (4K)', description: 'Maximum quality, slow processing' }
  ];

  const videoFormatOptions: SelectOption[] = [
    { value: 'mp4', label: 'MP4', description: 'Most compatible format' },
    { value: 'webm', label: 'WebM', description: 'Web-optimized format' },
    { value: 'avi', label: 'AVI', description: 'Uncompressed, large files' },
    { value: 'mov', label: 'MOV', description: 'Apple QuickTime format' }
  ];

  const frameRateOptions: SelectOption[] = [
    { value: 24, label: '24 FPS', description: 'Cinematic standard' },
    { value: 30, label: '30 FPS', description: 'Broadcast standard' },
    { value: 60, label: '60 FPS', description: 'Smooth motion' }
  ];

  return (
    <div className="space-y-6">
      <Select
        label="Default Video Quality"
        value={preferences.defaultVideoQuality}
        onValueChange={(value) => updatePreference('defaultVideoQuality', value as any)}
        options={videoQualityOptions}
      />
      
      <Select
        label="Default Video Format"
        value={preferences.defaultVideoFormat}
        onValueChange={(value) => updatePreference('defaultVideoFormat', value as any)}
        options={videoFormatOptions}
      />
      
      <Select
        label="Default Frame Rate"
        value={preferences.defaultFrameRate}
        onValueChange={(value) => updatePreference('defaultFrameRate', value as any)}
        options={frameRateOptions}
      />
    </div>
  );
} 