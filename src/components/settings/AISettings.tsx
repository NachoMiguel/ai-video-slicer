'use client';

import { Select, SelectOption } from '../ui/select';
import { Slider } from '../ui/slider';
import { Switch } from '../ui/switch';
import { useSettingsStore } from '../../stores/settingsStore';
import { Zap } from 'lucide-react';

export function AISettings() {
  const { preferences, updatePreference } = useSettingsStore();

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
    { value: 'short', label: 'Short (1-3 min)', description: 'Quick, concise content' },
    { value: 'medium', label: 'Medium (3-7 min)', description: 'Balanced depth and brevity' },
    { value: 'long', label: 'Long (7-15 min)', description: 'Comprehensive, detailed content' }
  ];

  return (
    <div className="space-y-6">
      <Select
        label="AI Model"
        value={preferences.preferredAIModel}
        onValueChange={(value) => updatePreference('preferredAIModel', value as any)}
        options={aiModelOptions}
      />
      
      <Select
        label="Prompt Style"
        value={preferences.defaultPromptStyle}
        onValueChange={(value) => updatePreference('defaultPromptStyle', value as any)}
        options={promptStyleOptions}
      />
      
      <Select
        label="Default Script Length"
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
      
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <label className="text-sm font-medium">Skip Character Extraction</label>
            <p className="text-xs text-muted-foreground">
              Use predefined characters instead of AI analysis for faster processing
            </p>
          </div>
          <Switch
            checked={preferences.skipCharacterExtraction}
            onCheckedChange={(checked) => updatePreference('skipCharacterExtraction', checked)}
          />
        </div>
        
        {preferences.skipCharacterExtraction && (
          <div className="mt-3 p-3 rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800">
            <div className="flex items-center gap-2 text-amber-800 dark:text-amber-200">
              <Zap className="h-4 w-4" />
              <span className="text-sm font-medium">
                Using predefined characters: Steven Seagal, Jean-Claude Van Damme
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
} 