'use client';

import { Card, CardContent, CardHeader, CardTitle } from '../../ui/card';
import { Button } from '../../ui/button';
import { 
  FileText, 
  BarChart3, 
  Clock, 
  Target, 
  TrendingUp, 
  AlertCircle, 
  CheckCircle, 
  Eye, 
  Lightbulb,
  Star,
  Users
} from 'lucide-react';

interface ScriptAnalysis {
  wordCount: number;
  readingTime: string;
  sections: number;
  overallScore: number;
  readabilityScore: number;
  engagementScore: number;
  structureScore: number;
  qualityLevel: 'excellent' | 'good' | 'fair' | 'poor';
  suggestions: {
    id: string;
    type: 'improvement' | 'warning' | 'tip';
    title: string;
    description: string;
    section?: string;
    priority: 'high' | 'medium' | 'low';
  }[];
  sectionsAnalysis: {
    name: string;
    wordCount: number;
    score: number;
    status: 'complete' | 'partial' | 'missing';
    suggestions: string[];
  }[];
  demographics: {
    targetAudience: string;
    complexity: 'beginner' | 'intermediate' | 'advanced';
    tone: 'casual' | 'professional' | 'energetic' | 'mixed';
  };
}

interface ScriptAnalysisDisplayProps {
  analysis: ScriptAnalysis;
  onRefineSection?: (sectionName: string) => void;
  onImplementSuggestion?: (suggestionId: string) => void;
  isLoading?: boolean;
}

export function ScriptAnalysisDisplay({ 
  analysis, 
  onRefineSection, 
  onImplementSuggestion,
  isLoading = false 
}: ScriptAnalysisDisplayProps) {
  
  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600 bg-green-100 dark:bg-green-900 dark:text-green-400';
    if (score >= 60) return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900 dark:text-yellow-400';
    return 'text-red-600 bg-red-100 dark:bg-red-900 dark:text-red-400';
  };

  const getQualityIcon = (level: string) => {
    switch (level) {
      case 'excellent':
        return <Star className="h-4 w-4 text-green-600" />;
      case 'good':
        return <CheckCircle className="h-4 w-4 text-blue-600" />;
      case 'fair':
        return <AlertCircle className="h-4 w-4 text-yellow-600" />;
      case 'poor':
        return <AlertCircle className="h-4 w-4 text-red-600" />;
      default:
        return <FileText className="h-4 w-4" />;
    }
  };

  const getSuggestionIcon = (type: string) => {
    switch (type) {
      case 'improvement':
        return <TrendingUp className="h-4 w-4 text-blue-500" />;
      case 'warning':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      case 'tip':
        return <Lightbulb className="h-4 w-4 text-yellow-500" />;
      default:
        return <Lightbulb className="h-4 w-4" />;
    }
  };

  const getPriorityBadgeColor = (priority: string) => {
    switch (priority) {
      case 'high':
        return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
      case 'low':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200';
    }
  };

  const renderMetricCard = (title: string, value: string | number, icon: React.ReactNode, description?: string) => (
    <Card className="text-center">
      <CardContent className="p-4">
        <div className="flex items-center justify-center mb-2">
          {icon}
        </div>
        <div className="text-2xl font-bold text-foreground">{value}</div>
        <div className="text-sm font-medium text-foreground">{title}</div>
        {description && (
          <div className="text-xs text-muted-foreground mt-1">{description}</div>
        )}
      </CardContent>
    </Card>
  );

  const renderScoreCard = (title: string, score: number, icon: React.ReactNode) => (
    <Card className="text-center">
      <CardContent className="p-4">
        <div className="flex items-center justify-center mb-2">
          {icon}
        </div>
        <div className={`text-2xl font-bold ${getScoreColor(score)}`}>
          {score}%
        </div>
        <div className="text-sm font-medium text-foreground">{title}</div>
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mt-2">
          <div 
            className={`h-2 rounded-full transition-all duration-300 ${
              score >= 80 ? 'bg-green-500' : score >= 60 ? 'bg-yellow-500' : 'bg-red-500'
            }`}
            style={{ width: `${score}%` }}
          />
        </div>
      </CardContent>
    </Card>
  );

  return (
    <div className="space-y-6">
      {/* Overall Quality Header */}
      <Card className={`border-2 ${
        analysis.qualityLevel === 'excellent' ? 'border-green-200 bg-green-50 dark:bg-green-950 dark:border-green-800' :
        analysis.qualityLevel === 'good' ? 'border-blue-200 bg-blue-50 dark:bg-blue-950 dark:border-blue-800' :
        analysis.qualityLevel === 'fair' ? 'border-yellow-200 bg-yellow-50 dark:bg-yellow-950 dark:border-yellow-800' :
        'border-red-200 bg-red-50 dark:bg-red-950 dark:border-red-800'
      }`}>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              {getQualityIcon(analysis.qualityLevel)}
              Script Analysis Overview
            </CardTitle>
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${
              analysis.qualityLevel === 'excellent' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
              analysis.qualityLevel === 'good' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200' :
              analysis.qualityLevel === 'fair' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
              'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
            }`}>
              {analysis.qualityLevel.charAt(0).toUpperCase() + analysis.qualityLevel.slice(1)} Quality
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <p className="text-muted-foreground">
                Your script has an overall quality score of <strong>{analysis.overallScore}%</strong> based on 
                readability, engagement, and structure analysis.
              </p>
            </div>
            <div className="text-right">
              <div className="text-3xl font-bold text-foreground">{analysis.overallScore}%</div>
              <div className="text-sm text-muted-foreground">Overall Score</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {renderMetricCard(
          'Words',
          analysis.wordCount.toLocaleString(),
          <FileText className="h-6 w-6 text-blue-500" />,
          'Total word count'
        )}
        {renderMetricCard(
          'Reading Time',
          analysis.readingTime,
          <Clock className="h-6 w-6 text-green-500" />,
          'Estimated time'
        )}
        {renderMetricCard(
          'Sections',
          analysis.sections,
          <BarChart3 className="h-6 w-6 text-purple-500" />,
          'Content sections'
        )}
        {renderMetricCard(
          'Audience',
          analysis.demographics.targetAudience,
          <Users className="h-6 w-6 text-orange-500" />,
          `${analysis.demographics.complexity} level`
        )}
      </div>

      {/* Detailed Scores */}
      <Card>
        <CardHeader>
          <CardTitle>Performance Breakdown</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {renderScoreCard(
              'Readability',
              analysis.readabilityScore,
              <Eye className="h-6 w-6" />
            )}
            {renderScoreCard(
              'Engagement',
              analysis.engagementScore,
              <TrendingUp className="h-6 w-6" />
            )}
            {renderScoreCard(
              'Structure',
              analysis.structureScore,
              <Target className="h-6 w-6" />
            )}
          </div>
        </CardContent>
      </Card>

      {/* Sections Analysis */}
      <Card>
        <CardHeader>
          <CardTitle>Section Breakdown</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {analysis.sectionsAnalysis.map((section, index) => (
              <div key={index} className="border border-border rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-3">
                    <h4 className="font-medium text-foreground">{section.name}</h4>
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      section.status === 'complete' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
                      section.status === 'partial' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
                      'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                    }`}>
                      {section.status}
                    </span>
                  </div>
                  
                  <div className="flex items-center gap-3">
                    <span className="text-sm text-muted-foreground">
                      {section.wordCount} words
                    </span>
                    <span className={`text-sm font-medium ${getScoreColor(section.score)}`}>
                      {section.score}%
                    </span>
                    {onRefineSection && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => onRefineSection(section.name)}
                        disabled={isLoading}
                      >
                        Refine
                      </Button>
                    )}
                  </div>
                </div>
                
                {section.suggestions.length > 0 && (
                  <div className="mt-2">
                    <p className="text-xs text-muted-foreground mb-1">Suggestions:</p>
                    <ul className="text-xs space-y-1">
                      {section.suggestions.map((suggestion, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-muted-foreground">
                          <span className="text-yellow-500 mt-0.5">-</span>
                          {suggestion}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Improvement Suggestions */}
      {analysis.suggestions.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Lightbulb className="h-5 w-5" />
              Improvement Suggestions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {analysis.suggestions.map((suggestion) => (
                <div 
                  key={suggestion.id}
                  className="border border-border rounded-lg p-4 hover:bg-accent/50 transition-colors"
                >
                  <div className="flex items-start gap-3">
                    <div className="flex-shrink-0 mt-0.5">
                      {getSuggestionIcon(suggestion.type)}
                    </div>
                    
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <h4 className="font-medium text-foreground">{suggestion.title}</h4>
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          getPriorityBadgeColor(suggestion.priority)
                        }`}>
                          {suggestion.priority}
                        </span>
                        {suggestion.section && (
                          <span className="text-xs text-muted-foreground bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">
                            {suggestion.section}
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-muted-foreground">{suggestion.description}</p>
                    </div>
                    
                    {onImplementSuggestion && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => onImplementSuggestion(suggestion.id)}
                        disabled={isLoading}
                        className="flex-shrink-0"
                      >
                        Apply
                      </Button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Demographics Info */}
      <Card className="bg-gray-50 dark:bg-gray-900">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            Content Profile
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-sm text-muted-foreground">Target Audience</div>
              <div className="font-medium text-foreground">{analysis.demographics.targetAudience}</div>
            </div>
            <div>
              <div className="text-sm text-muted-foreground">Complexity Level</div>
              <div className="font-medium text-foreground capitalize">{analysis.demographics.complexity}</div>
            </div>
            <div>
              <div className="text-sm text-muted-foreground">Tone</div>
              <div className="font-medium text-foreground capitalize">{analysis.demographics.tone}</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
} 