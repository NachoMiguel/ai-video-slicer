'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../../ui/card';
import { Button } from '../../ui/button';
import { Youtube, Clock, Star, Zap, ArrowRight, Play } from 'lucide-react';

interface EntryMethodSelectorProps {
  onMethodSelected: (method: 'youtube') => void;
  isLoading?: boolean;
}

export function EntryMethodSelector({ onMethodSelected, isLoading = false }: EntryMethodSelectorProps) {
  const handleContinue = () => {
    onMethodSelected('youtube');
  };

  return (
    <div className="w-full max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center space-y-4">
        <div className="flex items-center justify-center gap-3 mb-4">
          <div className="p-3 rounded-full bg-primary/10">
            <Youtube className="h-8 w-8 text-primary" />
          </div>
        </div>
        <h2 className="text-4xl font-bold text-foreground">
          YouTube Script Builder
        </h2>
        <p className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
          Transform any YouTube video into an optimized script with AI-powered analysis, 
          interactive refinement, and professional content enhancement.
        </p>
      </div>

      {/* Main Action Card */}
      <Card className="relative transition-all duration-300 hover:shadow-xl border-primary/20 bg-gradient-to-br from-primary/5 to-primary/10">
        {/* Premium Badge */}
        <div className="absolute -top-3 left-6 z-10">
          <div className="bg-gradient-to-r from-primary to-primary/80 text-primary-foreground px-4 py-2 rounded-full text-sm font-semibold flex items-center gap-2 shadow-lg">
            <Star className="h-3 w-3" />
            AI-Powered Workflow
          </div>
        </div>
        
        <CardHeader className="pb-6">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-4">
              <div className="p-4 rounded-xl bg-gradient-to-br from-primary to-primary/80 text-primary-foreground shadow-lg">
                <Youtube className="h-8 w-8" />
              </div>
              <div>
                <CardTitle className="text-2xl mb-2">Start with YouTube Video</CardTitle>
                <div className="flex items-center gap-3">
                  <span className="px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                    Simple & Fast
                  </span>
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <Clock className="h-4 w-4" />
                    <span className="text-sm">5-10 minutes</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </CardHeader>
        
        <CardContent className="space-y-6">
          <p className="text-lg text-muted-foreground leading-relaxed">
            Simply paste a YouTube URL and let our AI extract, analyze, and transform the content 
            into a professional script ready for video production.
          </p>
          
          {/* Workflow Steps */}
          <div className="space-y-4">
            <h4 className="font-semibold text-foreground flex items-center gap-2">
              <Play className="h-4 w-4 text-primary" />
              How it works:
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="flex items-start gap-3 p-3 rounded-lg bg-background/50 border">
                <div className="w-6 h-6 rounded-full bg-primary text-primary-foreground text-xs font-bold flex items-center justify-center flex-shrink-0 mt-0.5">
                  1
                </div>
                <div>
                  <h5 className="font-medium text-sm">Paste YouTube URL</h5>
                  <p className="text-xs text-muted-foreground mt-1">Enter any YouTube video link</p>
                </div>
              </div>
              <div className="flex items-start gap-3 p-3 rounded-lg bg-background/50 border">
                <div className="w-6 h-6 rounded-full bg-primary text-primary-foreground text-xs font-bold flex items-center justify-center flex-shrink-0 mt-0.5">
                  2
                </div>
                <div>
                  <h5 className="font-medium text-sm">AI Analysis</h5>
                  <p className="text-xs text-muted-foreground mt-1">Extract transcript & generate bullet points</p>
                </div>
              </div>
              <div className="flex items-start gap-3 p-3 rounded-lg bg-background/50 border">
                <div className="w-6 h-6 rounded-full bg-primary text-primary-foreground text-xs font-bold flex items-center justify-center flex-shrink-0 mt-0.5">
                  3
                </div>
                <div>
                  <h5 className="font-medium text-sm">Interactive Building</h5>
                  <p className="text-xs text-muted-foreground mt-1">Refine & enhance with AI assistance</p>
                </div>
              </div>
            </div>
          </div>
          
          {/* Features List */}
          <div className="space-y-3">
            <h4 className="font-semibold text-foreground">What you get:</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Zap className="h-3 w-3 text-primary flex-shrink-0" />
                Automatic transcript extraction
              </div>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Zap className="h-3 w-3 text-primary flex-shrink-0" />
                AI-powered content optimization
              </div>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Zap className="h-3 w-3 text-primary flex-shrink-0" />
                Interactive script refinement
              </div>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Zap className="h-3 w-3 text-primary flex-shrink-0" />
                Professional formatting
              </div>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Zap className="h-3 w-3 text-primary flex-shrink-0" />
                Engagement optimization
              </div>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Zap className="h-3 w-3 text-primary flex-shrink-0" />
                Real-time word count tracking
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Call to Action */}
      <div className="text-center space-y-4">
        <Button
          onClick={handleContinue}
          disabled={isLoading}
          size="lg"
          className="px-12 py-4 text-lg font-semibold bg-gradient-to-r from-primary to-primary/80 hover:from-primary/90 hover:to-primary/70 shadow-lg"
        >
          {isLoading ? (
            <div className="flex items-center gap-3">
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
              <span>Starting...</span>
            </div>
          ) : (
            <div className="flex items-center gap-3">
              <span>Start Building Your Script</span>
              <ArrowRight className="h-5 w-5" />
            </div>
          )}
        </Button>
        
        <p className="text-sm text-muted-foreground">
          No account required • Free to use • Privacy-focused
        </p>
      </div>
    </div>
  );
} 