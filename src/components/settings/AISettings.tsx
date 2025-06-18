'use client';

import { Select, SelectOption } from '../ui/select';
import { Slider } from '../ui/slider';
import { useSettingsStore } from '../../stores/settingsStore';

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
    { value: 'short', label: 'Short (200-400 words)', description: '1-2 minute videos' },
    { value: 'medium', label: 'Medium (400-800 words)', description: '3-5 minute videos' },
    { value: 'long', label: 'Long (800-1500 words)', description: '5-10 minute videos' },
    { value: 'custom', label: 'Custom Length', description: 'Specify exact word count' }
  ];

  return (
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
} 