// Script storage types
export interface SavedScript {
  id: string;
  title: string;
  word_count: number;
  created_at: string;
  updated_at: string;
  source_url?: string;
  filename: string;
  has_bullet_points: boolean;
  script_length: number;
  script_text?: string;
  bullet_points?: any[];
  ready_for_processing?: boolean;
}

export interface ScriptStorageState {
  scripts: SavedScript[];
  isLoading: boolean;
  error: string | null;
  selectedScript: SavedScript | null;
}

// Settings and preferences types
export interface UserPreferences {
  // Video Settings
  defaultVideoQuality: 'low' | 'medium' | 'high' | 'ultra';
  defaultVideoFormat: 'mp4' | 'webm' | 'avi' | 'mov';
  defaultFrameRate: 24 | 30 | 60;
  
  // AI Settings
  preferredAIModel: 'gpt-3.5-turbo' | 'gpt-4' | 'gpt-4-turbo';
  defaultPromptStyle: 'casual' | 'professional' | 'educational' | 'marketing';
  scriptLengthPreference: 'short' | 'medium' | 'long' | 'custom';
  customScriptLength?: number; // in words
  
  // UI Settings
  theme: 'light' | 'dark' | 'system';
  compactMode: boolean;
  showAdvancedFeatures: boolean;
  enableKeyboardShortcuts: boolean;
  
  // Export Settings
  defaultExportFormat: 'script' | 'srt' | 'pdf' | 'json';
  includeTimestamps: boolean;
  includeMetadata: boolean;
  
  // Performance Settings
  enableAutoSave: boolean;
  autoSaveInterval: number; // in seconds
  maxSessionHistory: number;
  enableAnalytics: boolean;
  
  // Accessibility
  enableHighContrast: boolean;
  fontSize: 'small' | 'medium' | 'large' | 'xlarge';
  enableScreenReader: boolean;
  reducedMotion: boolean;
}

export interface AppSettings {
  preferences: UserPreferences;
  lastModified: string;
  version: string;
}

export interface SettingsSection {
  id: string;
  title: string;
  description: string;
  icon: string;
  items: SettingsItem[];
}

export interface SettingsItem {
  id: keyof UserPreferences;
  label: string;
  description: string;
  type: 'select' | 'toggle' | 'slider' | 'input' | 'custom';
  options?: Array<{ value: any; label: string; description?: string }>;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  validation?: (value: any) => boolean | string;
}

// Default settings
export const DEFAULT_PREFERENCES: UserPreferences = {
  // Video Settings
  defaultVideoQuality: 'high',
  defaultVideoFormat: 'mp4',
  defaultFrameRate: 30,
  
  // AI Settings
  preferredAIModel: 'gpt-4-turbo',
  defaultPromptStyle: 'professional',
  scriptLengthPreference: 'medium',
  customScriptLength: 500,
  
  // UI Settings
  theme: 'system',
  compactMode: false,
  showAdvancedFeatures: true,
  enableKeyboardShortcuts: true,
  
  // Export Settings
  defaultExportFormat: 'script',
  includeTimestamps: true,
  includeMetadata: true,
  
  // Performance Settings
  enableAutoSave: true,
  autoSaveInterval: 30,
  maxSessionHistory: 10,
  enableAnalytics: false,
  
  // Accessibility
  enableHighContrast: false,
  fontSize: 'medium',
  enableScreenReader: false,
  reducedMotion: false,
};

// Settings validation
export const validateSettings = (preferences: Partial<UserPreferences>): string[] => {
  const errors: string[] = [];
  
  if (preferences.customScriptLength && (preferences.customScriptLength < 10 || preferences.customScriptLength > 10000)) {
    errors.push('Custom script length must be between 10 and 10,000 words');
  }
  
  if (preferences.autoSaveInterval && (preferences.autoSaveInterval < 5 || preferences.autoSaveInterval > 300)) {
    errors.push('Auto-save interval must be between 5 and 300 seconds');
  }
  
  if (preferences.maxSessionHistory && (preferences.maxSessionHistory < 1 || preferences.maxSessionHistory > 50)) {
    errors.push('Session history limit must be between 1 and 50');
  }
  
  return errors;
}; 