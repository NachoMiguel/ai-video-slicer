'use client';

import { useState } from 'react';
import { Button } from '../../ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../../ui/card';
import { 
  Check, 
  X, 
  Eye, 
  EyeOff, 
  Loader2,
  CheckCircle,
  XCircle,
  AlertCircle
} from 'lucide-react';

interface BulkModificationResult {
  index: number;
  original_text: string;
  modified_text: string;
  start_offset: number;
  end_offset: number;
  success: boolean;
  error?: string;
}

interface BulkModificationPreviewProps {
  visible: boolean;
  modificationType: string;
  results: BulkModificationResult[];
  isApplying: boolean;
  onApply: (selectedResults: BulkModificationResult[]) => void;
  onCancel: () => void;
}

export function BulkModificationPreview({
  visible,
  modificationType,
  results,
  isApplying,
  onApply,
  onCancel
}: BulkModificationPreviewProps) {
  const [selectedResults, setSelectedResults] = useState<Set<number>>(
    new Set(results.filter(r => r.success).map(r => r.index))
  );
  const [expandedItems, setExpandedItems] = useState<Set<number>>(new Set());

  if (!visible) return null;

  const handleToggleSelection = (index: number) => {
    const newSelected = new Set(selectedResults);
    if (newSelected.has(index)) {
      newSelected.delete(index);
    } else {
      newSelected.add(index);
    }
    setSelectedResults(newSelected);
  };

  const handleToggleExpanded = (index: number) => {
    const newExpanded = new Set(expandedItems);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedItems(newExpanded);
  };

  const handleSelectAll = () => {
    const successfulResults = results.filter(r => r.success);
    setSelectedResults(new Set(successfulResults.map(r => r.index)));
  };

  const handleSelectNone = () => {
    setSelectedResults(new Set());
  };

  const handleApply = () => {
    const selectedResultsArray = results.filter(r => selectedResults.has(r.index));
    onApply(selectedResultsArray);
  };

  const successfulResults = results.filter(r => r.success);
  const failedResults = results.filter(r => !r.success);
  const selectedCount = selectedResults.size;

  const getModificationTypeLabel = (type: string) => {
    const labels = {
      'shorten': 'Shorten',
      'expand': 'Expand', 
      'rewrite': 'Rewrite',
      'make_engaging': 'Make Engaging',
      'delete': 'Delete'
    };
    return labels[type as keyof typeof labels] || type;
  };

  const truncateText = (text: string, maxLength: number = 100) => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 animate-in fade-in-0 duration-200">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-4xl w-full mx-4 max-h-[90vh] overflow-hidden flex flex-col">
        <CardHeader className="flex-shrink-0 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg font-semibold text-foreground">
              Bulk Modification Preview: {getModificationTypeLabel(modificationType)}
            </CardTitle>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <span>{selectedCount} of {successfulResults.length} selected</span>
              {failedResults.length > 0 && (
                <span className="text-red-500">({failedResults.length} failed)</span>
              )}
            </div>
          </div>
          
          {/* Selection Controls */}
          <div className="flex items-center gap-2 mt-3">
            <Button
              variant="outline"
              size="sm"
              onClick={handleSelectAll}
              disabled={isApplying}
            >
              Select All ({successfulResults.length})
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleSelectNone}
              disabled={isApplying}
            >
              Select None
            </Button>
          </div>
        </CardHeader>
        
        <CardContent className="flex-1 overflow-y-auto p-0">
          <div className="space-y-3 p-4">
            {results.map((result, index) => {
              const isSelected = selectedResults.has(result.index);
              const isExpanded = expandedItems.has(result.index);
              const isSuccess = result.success;
              
              return (
                <Card 
                  key={result.index}
                  className={`border transition-all duration-200 ${
                    isSuccess 
                      ? isSelected 
                        ? 'border-green-300 bg-green-50 dark:bg-green-950 dark:border-green-700' 
                        : 'border-gray-200 dark:border-gray-700'
                      : 'border-red-300 bg-red-50 dark:bg-red-950 dark:border-red-700'
                  }`}
                >
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        {isSuccess ? (
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleToggleSelection(result.index)}
                            disabled={isApplying}
                            className={`h-8 w-8 p-0 ${
                              isSelected 
                                ? 'text-green-600 hover:text-green-700' 
                                : 'text-gray-400 hover:text-gray-600'
                            }`}
                          >
                            {isSelected ? <CheckCircle className="h-4 w-4" /> : <div className="h-4 w-4 border border-gray-300 rounded-full" />}
                          </Button>
                        ) : (
                          <XCircle className="h-4 w-4 text-red-500" />
                        )}
                        
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium">
                            Selection {result.index + 1}
                          </span>
                          {!isSuccess && (
                            <div className="flex items-center gap-1 text-xs text-red-600">
                              <AlertCircle className="h-3 w-3" />
                              Failed
                            </div>
                          )}
                        </div>
                      </div>
                      
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleToggleExpanded(result.index)}
                        className="h-8 w-8 p-0"
                      >
                        {isExpanded ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                      </Button>
                    </div>
                  </CardHeader>
                  
                  <CardContent className="pt-0">
                    {isExpanded ? (
                      <div className="space-y-3">
                        <div>
                          <h4 className="text-xs font-medium text-muted-foreground mb-1">Original Text:</h4>
                          <div className="bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 rounded p-2 text-sm">
                            {result.original_text}
                          </div>
                        </div>
                        
                        <div>
                          <h4 className="text-xs font-medium text-muted-foreground mb-1">
                            {isSuccess ? 'Modified Text:' : 'Error:'}
                          </h4>
                          <div className={`border rounded p-2 text-sm ${
                            isSuccess 
                              ? 'bg-green-50 dark:bg-green-950 border-green-200 dark:border-green-800'
                              : 'bg-red-50 dark:bg-red-950 border-red-200 dark:border-red-800 text-red-700 dark:text-red-300'
                          }`}>
                            {isSuccess ? result.modified_text : result.error}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <div className="text-xs text-muted-foreground">
                          Original: {truncateText(result.original_text)}
                        </div>
                        {isSuccess && (
                          <div className="text-xs text-muted-foreground">
                            Modified: {truncateText(result.modified_text)}
                          </div>
                        )}
                        {!isSuccess && (
                          <div className="text-xs text-red-600">
                            Error: {truncateText(result.error || 'Unknown error')}
                          </div>
                        )}
                      </div>
                    )}
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </CardContent>
        
        {/* Footer Actions */}
        <div className="flex-shrink-0 border-t border-gray-200 dark:border-gray-700 p-4">
          <div className="flex items-center justify-between">
            <div className="text-sm text-muted-foreground">
              {selectedCount > 0 ? (
                `Ready to apply ${selectedCount} modification${selectedCount === 1 ? '' : 's'}`
              ) : (
                'Select modifications to apply'
              )}
            </div>
            
            <div className="flex items-center gap-3">
              <Button
                variant="outline"
                onClick={onCancel}
                disabled={isApplying}
              >
                Cancel
              </Button>
              <Button
                onClick={handleApply}
                disabled={isApplying || selectedCount === 0}
                className="bg-green-600 hover:bg-green-700"
              >
                {isApplying ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Applying {selectedCount} changes...
                  </>
                ) : (
                  `Apply ${selectedCount} Change${selectedCount === 1 ? '' : 's'}`
                )}
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 