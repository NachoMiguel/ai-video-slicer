// Main component
export { ScriptBuilder } from './ScriptBuilder';

// Entry components
export { EntryMethodSelector } from './entry/EntryMethodSelector';
export { YouTubeInputPanel } from './entry/YouTubeInputPanel';

// Interactive components
export { ChatInterface } from './interactive/ChatInterface';

// Shared components
export { ScriptAnalysisDisplay } from './shared/ScriptAnalysisDisplay';
export { WordCountTracker } from './shared/WordCountTracker';
export { BulkModificationPreview } from './shared/BulkModificationPreview';

// Hooks
export { useBulkSelection } from '../../hooks/useBulkSelection';
export type { BulkSelection } from '../../hooks/useBulkSelection';

// Types (you might want to add these later for external use)
export type { ChatMessage, ScriptSession } from './ScriptBuilder'; 