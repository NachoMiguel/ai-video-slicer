'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Switch } from '../ui/switch';
import { Select, SelectOption } from '../ui/select';
import { Slider } from '../ui/slider';
import { useSettingsStore } from '../../stores/settingsStore';
import { VideoSettings } from './VideoSettings';
import { AISettings } from './AISettings';
import { 
  Settings, 
  Video, 
  Bot, 
  Palette, 
  Download,
  Zap,
  Eye,
  Save,
  RotateCcw,
  AlertCircle,
  CheckCircle,
  X
} from 'lucide-react';

interface SettingsPanelProps {
  onClose?: () => void;
  className?: string;
}

export function SettingsPanel({ onClose, className = '' }: SettingsPanelProps) {
  const {
    preferences,
    updatePreference,
    resetToDefaults,
    resetSection,
    save,
    getValidationErrors,
    hasUnsavedChanges
  } = useSettingsStore();

  const [activeSection, setActiveSection] = useState('video');
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');
  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  const handleSave = async () => {
    setSaveStatus('saving');
    try {
      const errors = await save();
      if (errors.length > 0) {
        setValidationErrors(errors);
        setSaveStatus('error');
      } else {
        setValidationErrors([]);
        setSaveStatus('success');
        setTimeout(() => setSaveStatus('idle'), 2000);
      }
    } catch (error) {
      setSaveStatus('error');
      setValidationErrors(['Failed to save settings']);
    }
  };

  const handleResetSection = () => {
    resetSection(activeSection);
  };

  const sections = [
    {
      id: 'video',
      title: 'Video Settings',
      icon: Video,
      description: 'Default video quality and format preferences'
    },
    {
      id: 'ai',
      title: 'AI Settings',
      icon: Bot,
      description: 'AI model preferences and script generation settings'
    },
    {
      id: 'ui',
      title: 'Interface',
      icon: Palette,
      description: 'Theme, layout, and display preferences'
    },
    {
      id: 'export',
      title: 'Export',
      icon: Download,
      description: 'Default export formats and options'
    },
    {
      id: 'performance',
      title: 'Performance',
      icon: Zap,
      description: 'Auto-save, caching, and performance settings'
    },
    {
      id: 'accessibility',
      title: 'Accessibility',
      icon: Eye,
      description: 'Accessibility and display accommodations'
    }
  ];

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

  const aiModelOptions: SelectOption[] = [
    { value: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo', description: 'Fast and cost-effective' },
    { value: 'gpt-4', label: 'GPT-4', description: 'Higher quality, slower' },
    { value: 'gpt-4-turbo', label: 'GPT-4 Turbo', description: 'Best quality and speed balance' }
  ];

  const promptStyleOptions: SelectOption[] = [
    { value: 'casual', label: 'Casual', description: 'Conversational and friendly' },
    { value: 'professional', label: 'Professional', description: 'Formal business tone' },
    { value: 'educational', label: 'Educational', description: 'Informative and clear' },
    { value: 'marketing', label: 'Marketing', description: 'Persuasive and engaging' }
  ];

  const scriptLengthOptions: SelectOption[] = [
    { value: 'short', label: 'Short (200-400 words)', description: '1-2 minute videos' },
    { value: 'medium', label: 'Medium (400-800 words)', description: '3-5 minute videos' },
    { value: 'long', label: 'Long (800-1500 words)', description: '5-10 minute videos' },
    { value: 'custom', label: 'Custom Length', description: 'Specify exact word count' }
  ];

  const themeOptions: SelectOption[] = [
    { value: 'light', label: 'Light', description: 'Light theme' },
    { value: 'dark', label: 'Dark', description: 'Dark theme' },
    { value: 'system', label: 'System', description: 'Follow system preference' }
  ];

  const exportFormatOptions: SelectOption[] = [
    { value: 'script', label: 'Plain Text Script', description: 'Simple text file' },
    { value: 'srt', label: 'SRT Subtitles', description: 'Subtitle file with timestamps' },
    { value: 'pdf', label: 'PDF Document', description: 'Formatted document' },
    { value: 'json', label: 'JSON Data', description: 'Structured data format' }
  ];

  const fontSizeOptions: SelectOption[] = [
    { value: 'small', label: 'Small', description: '14px base size' },
    { value: 'medium', label: 'Medium', description: '16px base size' },
    { value: 'large', label: 'Large', description: '18px base size' },
    { value: 'xlarge', label: 'Extra Large', description: '20px base size' }
  ];

  const renderVideoSettings = () => (
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

  const renderAISettings = () => (
    <div className="space-y-6">
      <Select
        label="Preferred AI Model"
        value={preferences.preferredAIModel}
        onValueChange={(value) => updatePreference('preferredAIModel', value as any)}
        options={aiModelOptions}
      />
      
      <Select
        label="Default Prompt Style"
        value={preferences.defaultPromptStyle}
        onValueChange={(value) => updatePreference('defaultPromptStyle', value as any)}
        options={promptStyleOptions}
      />
      
      <Select
        label="Script Length Preference"
        value={preferences.scriptLengthPreference}
        onValueChange={(value) => updatePreference('scriptLengthPreference', value as any)}
        options={scriptLengthOptions}
      />
      
      {preferences.scriptLengthPreference === 'custom' && (
        <Slider
          label="Custom Script Length"
          value={preferences.customScriptLength || 500}
          onValueChange={(value) => updatePreference('customScriptLength', value)}
          min={50}
          max={2000}
          step={50}
          unit="words"
        />
      )}
    </div>
  );

  const renderUISettings = () => (
    <div className="space-y-6">
      <Select
        label="Theme"
        value={preferences.theme}
        onValueChange={(value) => updatePreference('theme', value as any)}
        options={themeOptions}
      />
      
      <div className="flex items-center justify-between">
        <div>
          <label className="text-sm font-medium text-foreground">Compact Mode</label>
          <p className="text-xs text-muted-foreground">Reduce spacing and padding</p>
        </div>
        <Switch
          checked={preferences.compactMode}
          onCheckedChange={(checked) => updatePreference('compactMode', checked)}
        />
      </div>
      
      <div className="flex items-center justify-between">
        <div>
          <label className="text-sm font-medium text-foreground">Show Advanced Features</label>
          <p className="text-xs text-muted-foreground">Display power user options</p>
        </div>
        <Switch
          checked={preferences.showAdvancedFeatures}
          onCheckedChange={(checked) => updatePreference('showAdvancedFeatures', checked)}
        />
      </div>
      
      <div className="flex items-center justify-between">
        <div>
          <label className="text-sm font-medium text-foreground">Enable Keyboard Shortcuts</label>
          <p className="text-xs text-muted-foreground">Quick actions with hotkeys</p>
        </div>
        <Switch
          checked={preferences.enableKeyboardShortcuts}
          onCheckedChange={(checked) => updatePreference('enableKeyboardShortcuts', checked)}
        />
      </div>
    </div>
  );

  const renderExportSettings = () => (
    <div className="space-y-6">
      <Select
        label="Default Export Format"
        value={preferences.defaultExportFormat}
        onValueChange={(value) => updatePreference('defaultExportFormat', value as any)}
        options={exportFormatOptions}
      />
      
      <div className="flex items-center justify-between">
        <div>
          <label className="text-sm font-medium text-foreground">Include Timestamps</label>
          <p className="text-xs text-muted-foreground">Add timing information to exports</p>
        </div>
        <Switch
          checked={preferences.includeTimestamps}
          onCheckedChange={(checked) => updatePreference('includeTimestamps', checked)}
        />
      </div>
      
      <div className="flex items-center justify-between">
        <div>
          <label className="text-sm font-medium text-foreground">Include Metadata</label>
          <p className="text-xs text-muted-foreground">Add creation date and settings info</p>
        </div>
        <Switch
          checked={preferences.includeMetadata}
          onCheckedChange={(checked) => updatePreference('includeMetadata', checked)}
        />
      </div>
    </div>
  );

  const renderPerformanceSettings = () => (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <label className="text-sm font-medium text-foreground">Enable Auto-Save</label>
          <p className="text-xs text-muted-foreground">Automatically save work in progress</p>
        </div>
        <Switch
          checked={preferences.enableAutoSave}
          onCheckedChange={(checked) => updatePreference('enableAutoSave', checked)}
        />
      </div>
      
      {preferences.enableAutoSave && (
        <Slider
          label="Auto-Save Interval"
          value={preferences.autoSaveInterval}
          onValueChange={(value) => updatePreference('autoSaveInterval', value)}
          min={5}
          max={300}
          step={5}
          unit="seconds"
        />
      )}
      
      <Slider
        label="Session History Limit"
        value={preferences.maxSessionHistory}
        onValueChange={(value) => updatePreference('maxSessionHistory', value)}
        min={1}
        max={50}
        step={1}
        unit="sessions"
      />
      
      <div className="flex items-center justify-between">
        <div>
          <label className="text-sm font-medium text-foreground">Enable Analytics</label>
          <p className="text-xs text-muted-foreground">Help improve the app with usage data</p>
        </div>
        <Switch
          checked={preferences.enableAnalytics}
          onCheckedChange={(checked) => updatePreference('enableAnalytics', checked)}
        />
      </div>
    </div>
  );

  const renderAccessibilitySettings = () => (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <label className="text-sm font-medium text-foreground">High Contrast Mode</label>
          <p className="text-xs text-muted-foreground">Increase color contrast for better visibility</p>
        </div>
        <Switch
          checked={preferences.enableHighContrast}
          onCheckedChange={(checked) => updatePreference('enableHighContrast', checked)}
        />
      </div>
      
      <Select
        label="Font Size"
        value={preferences.fontSize}
        onValueChange={(value) => updatePreference('fontSize', value as any)}
        options={fontSizeOptions}
      />
      
      <div className="flex items-center justify-between">
        <div>
          <label className="text-sm font-medium text-foreground">Screen Reader Support</label>
          <p className="text-xs text-muted-foreground">Enhanced accessibility features</p>
        </div>
        <Switch
          checked={preferences.enableScreenReader}
          onCheckedChange={(checked) => updatePreference('enableScreenReader', checked)}
        />
      </div>
      
      <div className="flex items-center justify-between">
        <div>
          <label className="text-sm font-medium text-foreground">Reduced Motion</label>
          <p className="text-xs text-muted-foreground">Minimize animations and transitions</p>
        </div>
        <Switch
          checked={preferences.reducedMotion}
          onCheckedChange={(checked) => updatePreference('reducedMotion', checked)}
        />
      </div>
    </div>
  );

  const renderActiveSection = () => {
    switch (activeSection) {
      case 'video': return renderVideoSettings();
      case 'ai': return renderAISettings();
      case 'ui': return renderUISettings();
      case 'export': return renderExportSettings();
      case 'performance': return renderPerformanceSettings();
      case 'accessibility': return renderAccessibilitySettings();
      default: return null;
    }
  };

  return (
    <div className={`bg-background min-h-screen ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-6 border-b border-border">
        <div className="flex items-center gap-3">
          <Settings className="h-6 w-6 text-primary" />
          <div>
            <h1 className="text-2xl font-bold text-foreground">Settings</h1>
            <p className="text-muted-foreground">Customize your AI Video Slicer experience</p>
          </div>
        </div>
        {onClose && (
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        )}
      </div>

      <div className="flex min-h-0">
        {/* Sidebar */}
        <div className="w-64 border-r border-border p-4">
          <nav className="space-y-2">
            {sections.map((section) => {
              const IconComponent = section.icon;
              return (
                <button
                  key={section.id}
                  onClick={() => setActiveSection(section.id)}
                  className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left transition-colors ${
                    activeSection === section.id
                      ? 'bg-primary text-primary-foreground'
                      : 'text-foreground hover:bg-gray-100 dark:hover:bg-gray-800'
                  }`}
                >
                  <IconComponent className="h-4 w-4" />
                  <div>
                    <div className="font-medium">{section.title}</div>
                    <div className="text-xs opacity-75">{section.description}</div>
                  </div>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Main Content */}
        <div className="flex-1 p-6 overflow-y-auto">
          <div className="max-w-2xl">
            {/* Section Header */}
            <div className="mb-6">
              <h2 className="text-xl font-semibold text-foreground mb-2">
                {sections.find(s => s.id === activeSection)?.title}
              </h2>
              <p className="text-muted-foreground">
                {sections.find(s => s.id === activeSection)?.description}
              </p>
            </div>

            {/* Validation Errors */}
            {validationErrors.length > 0 && (
              <div className="mb-6 p-4 bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 rounded-lg">
                <div className="flex items-center gap-2 text-red-800 dark:text-red-200 mb-2">
                  <AlertCircle className="h-4 w-4" />
                  <span className="font-medium">Validation Errors</span>
                </div>
                <ul className="text-sm text-red-700 dark:text-red-300 space-y-1">
                  {validationErrors.map((error, index) => (
                    <li key={index}>• {error}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Settings Content */}
            <Card>
              <CardContent className="p-6">
                {renderActiveSection()}
              </CardContent>
            </Card>

            {/* Actions */}
            <div className="flex items-center justify-between mt-6">
              <Button
                variant="outline"
                onClick={handleResetSection}
                className="flex items-center gap-2"
              >
                <RotateCcw className="h-4 w-4" />
                Reset Section
              </Button>

              <div className="flex items-center gap-3">
                {hasUnsavedChanges() && (
                  <span className="text-sm text-muted-foreground">Unsaved changes</span>
                )}
                
                <Button
                  variant="outline"
                  onClick={resetToDefaults}
                >
                  Reset All
                </Button>
                
                <Button
                  onClick={handleSave}
                  disabled={saveStatus === 'saving'}
                  className="flex items-center gap-2"
                >
                  {saveStatus === 'saving' ? (
                    <>
                      <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full" />
                      Saving...
                    </>
                  ) : saveStatus === 'success' ? (
                    <>
                      <CheckCircle className="h-4 w-4" />
                      Saved
                    </>
                  ) : (
                    <>
                      <Save className="h-4 w-4" />
                      Save Settings
                    </>
                  )}
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 