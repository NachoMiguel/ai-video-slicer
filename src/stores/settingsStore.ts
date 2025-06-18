import { create } from 'zustand';
import { UserPreferences, DEFAULT_PREFERENCES, validateSettings } from '../types/settings';

interface SettingsStore {
  // State
  preferences: UserPreferences;
  isDirty: boolean;
  lastSaved: string | null;
  
  // Actions
  updatePreference: (key: keyof UserPreferences, value: any) => void;
  updatePreferences: (preferences: Partial<UserPreferences>) => void;
  resetToDefaults: () => void;
  resetSection: (section: string) => void;
  save: () => Promise<string[]>;
  
  // Computed values
  getValidationErrors: () => string[];
  hasUnsavedChanges: () => boolean;
}

const STORAGE_KEY = 'ai-video-slicer-settings';

// Load initial preferences from localStorage
const loadStoredPreferences = (): UserPreferences => {
  // Only access localStorage on the client side
  if (typeof window === "undefined") {
    return DEFAULT_PREFERENCES;
  }
  
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      return { ...DEFAULT_PREFERENCES, ...parsed };
    }
  } catch (error) {
    console.warn('Failed to load stored preferences:', error);
  }
  return DEFAULT_PREFERENCES;
};

// Save preferences to localStorage
const savePreferences = (preferences: UserPreferences) => {
  // Only access localStorage on the client side
  if (typeof window === "undefined") {
    return false;
  }
  
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(preferences));
    return true;
  } catch (error) {
    console.error('Failed to save preferences:', error);
    return false;
  }
};

export const useSettingsStore = create<SettingsStore>((set, get) => ({
  // Initial state
  preferences: loadStoredPreferences(),
  isDirty: false,
  lastSaved: null,
  
  // Actions
  updatePreference: (key, value) => {
    set((state) => ({
      preferences: {
        ...state.preferences,
        [key]: value,
      },
      isDirty: true,
    }));
  },
  
  updatePreferences: (newPreferences) => {
    set((state) => ({
      preferences: {
        ...state.preferences,
        ...newPreferences,
      },
      isDirty: true,
    }));
  },
  
  resetToDefaults: () => {
    set({
      preferences: { ...DEFAULT_PREFERENCES },
      isDirty: true,
    });
  },
  
  resetSection: (section) => {
    const state = get();
    let sectionDefaults: Partial<UserPreferences> = {};
    
    switch (section) {
      case 'video':
        sectionDefaults = {
          defaultVideoQuality: DEFAULT_PREFERENCES.defaultVideoQuality,
          defaultVideoFormat: DEFAULT_PREFERENCES.defaultVideoFormat,
          defaultFrameRate: DEFAULT_PREFERENCES.defaultFrameRate,
        };
        break;
      case 'ai':
        sectionDefaults = {
          preferredAIModel: DEFAULT_PREFERENCES.preferredAIModel,
          defaultPromptStyle: DEFAULT_PREFERENCES.defaultPromptStyle,
          scriptLengthPreference: DEFAULT_PREFERENCES.scriptLengthPreference,
          customScriptLength: DEFAULT_PREFERENCES.customScriptLength,
        };
        break;
      case 'ui':
        sectionDefaults = {
          theme: DEFAULT_PREFERENCES.theme,
          compactMode: DEFAULT_PREFERENCES.compactMode,
          showAdvancedFeatures: DEFAULT_PREFERENCES.showAdvancedFeatures,
          enableKeyboardShortcuts: DEFAULT_PREFERENCES.enableKeyboardShortcuts,
        };
        break;
      case 'export':
        sectionDefaults = {
          defaultExportFormat: DEFAULT_PREFERENCES.defaultExportFormat,
          includeTimestamps: DEFAULT_PREFERENCES.includeTimestamps,
          includeMetadata: DEFAULT_PREFERENCES.includeMetadata,
        };
        break;
      case 'performance':
        sectionDefaults = {
          enableAutoSave: DEFAULT_PREFERENCES.enableAutoSave,
          autoSaveInterval: DEFAULT_PREFERENCES.autoSaveInterval,
          maxSessionHistory: DEFAULT_PREFERENCES.maxSessionHistory,
          enableAnalytics: DEFAULT_PREFERENCES.enableAnalytics,
        };
        break;
      case 'accessibility':
        sectionDefaults = {
          enableHighContrast: DEFAULT_PREFERENCES.enableHighContrast,
          fontSize: DEFAULT_PREFERENCES.fontSize,
          enableScreenReader: DEFAULT_PREFERENCES.enableScreenReader,
          reducedMotion: DEFAULT_PREFERENCES.reducedMotion,
        };
        break;
    }
    
    state.updatePreferences(sectionDefaults);
  },
  
  save: async () => {
    const state = get();
    const errors = validateSettings(state.preferences);
    
    if (errors.length === 0) {
      const success = savePreferences(state.preferences);
      if (success) {
        set({
          isDirty: false,
          lastSaved: new Date().toISOString(),
        });
      }
    }
    
    return errors;
  },
  
  // Computed values
  getValidationErrors: () => {
    return validateSettings(get().preferences);
  },
  
  hasUnsavedChanges: () => {
    return get().isDirty;
  },
}));

// Utility hook for specific preference access
export const usePreference = <K extends keyof UserPreferences>(key: K): UserPreferences[K] => {
  return useSettingsStore((state) => state.preferences[key]);
};

// Utility hook for theme specifically (commonly used)
export const useTheme = () => {
  return useSettingsStore((state) => state.preferences.theme);
};

// Utility hook for checking if advanced features should be shown
export const useAdvancedFeatures = () => {
  return useSettingsStore((state) => state.preferences.showAdvancedFeatures);
}; 