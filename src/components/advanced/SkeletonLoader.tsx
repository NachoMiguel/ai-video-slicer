'use client';

import { ReactNode } from 'react';

interface SkeletonProps {
  className?: string;
  variant?: 'text' | 'rectangle' | 'circle' | 'button';
  width?: string | number;
  height?: string | number;
  animate?: boolean;
}

export function Skeleton({ 
  className = '', 
  variant = 'rectangle', 
  width, 
  height, 
  animate = true 
}: SkeletonProps) {
  const baseClasses = `bg-gray-200 dark:bg-gray-700 ${animate ? 'animate-pulse' : ''}`;
  
  const variantClasses = {
    text: 'h-4 rounded',
    rectangle: 'rounded-md',
    circle: 'rounded-full',
    button: 'h-10 rounded-lg'
  };

  const style: React.CSSProperties = {
    width: width || (variant === 'circle' ? '2.5rem' : '100%'),
    height: height || (variant === 'text' ? '1rem' : variant === 'button' ? '2.5rem' : '1.5rem')
  };

  return (
    <div 
      className={`${baseClasses} ${variantClasses[variant]} ${className}`}
      style={style}
    />
  );
}

interface SkeletonTextProps {
  lines?: number;
  className?: string;
}

export function SkeletonText({ lines = 3, className = '' }: SkeletonTextProps) {
  return (
    <div className={`space-y-2 ${className}`}>
      {Array.from({ length: lines }).map((_, index) => (
        <Skeleton 
          key={index}
          variant="text" 
          width={index === lines - 1 ? '75%' : '100%'}
        />
      ))}
    </div>
  );
}

interface SkeletonCardProps {
  showImage?: boolean;
  showAvatar?: boolean;
  lines?: number;
  className?: string;
}

export function SkeletonCard({ 
  showImage = false, 
  showAvatar = false, 
  lines = 3, 
  className = '' 
}: SkeletonCardProps) {
  return (
    <div className={`p-4 border border-border rounded-lg bg-card ${className}`}>
      {showImage && (
        <Skeleton variant="rectangle" height="12rem" className="mb-4" />
      )}
      
      <div className="space-y-3">
        {showAvatar && (
          <div className="flex items-center space-x-3">
            <Skeleton variant="circle" width="2.5rem" height="2.5rem" />
            <div className="flex-1">
              <Skeleton variant="text" width="40%" />
              <Skeleton variant="text" width="60%" className="mt-1" />
            </div>
          </div>
        )}
        
        <div className="space-y-2">
          <Skeleton variant="text" width="90%" height="1.25rem" />
          <SkeletonText lines={lines} />
        </div>
        
        <div className="flex space-x-2 pt-2">
          <Skeleton variant="button" width="5rem" />
          <Skeleton variant="button" width="4rem" />
        </div>
      </div>
    </div>
  );
}

interface SkeletonListProps {
  items?: number;
  showAvatar?: boolean;
  className?: string;
}

export function SkeletonList({ items = 5, showAvatar = true, className = '' }: SkeletonListProps) {
  return (
    <div className={`space-y-3 ${className}`}>
      {Array.from({ length: items }).map((_, index) => (
        <div key={index} className="flex items-center space-x-3 p-3 rounded-lg bg-card border border-border">
          {showAvatar && (
            <Skeleton variant="circle" width="2.5rem" height="2.5rem" />
          )}
          <div className="flex-1 space-y-2">
            <Skeleton variant="text" width="70%" />
            <Skeleton variant="text" width="50%" />
          </div>
          <Skeleton variant="button" width="4rem" />
        </div>
      ))}
    </div>
  );
}

interface SkeletonTableProps {
  rows?: number;
  columns?: number;
  showHeader?: boolean;
  className?: string;
}

export function SkeletonTable({ 
  rows = 5, 
  columns = 4, 
  showHeader = true, 
  className = '' 
}: SkeletonTableProps) {
  return (
    <div className={`border border-border rounded-lg overflow-hidden ${className}`}>
      {showHeader && (
        <div className="bg-gray-50 dark:bg-gray-800 p-4 border-b border-border">
          <div className="grid grid-cols-4 gap-4">
            {Array.from({ length: columns }).map((_, index) => (
              <Skeleton key={index} variant="text" width="60%" />
            ))}
          </div>
        </div>
      )}
      <div className="divide-y divide-border">
        {Array.from({ length: rows }).map((_, rowIndex) => (
          <div key={rowIndex} className="p-4">
            <div className="grid grid-cols-4 gap-4">
              {Array.from({ length: columns }).map((_, colIndex) => (
                <Skeleton key={colIndex} variant="text" />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

interface SkeletonChartProps {
  type?: 'bar' | 'line' | 'pie';
  className?: string;
}

export function SkeletonChart({ type = 'bar', className = '' }: SkeletonChartProps) {
  if (type === 'pie') {
    return (
      <div className={`flex items-center justify-center ${className}`}>
        <Skeleton variant="circle" width="12rem" height="12rem" />
      </div>
    );
  }

  if (type === 'line') {
    return (
      <div className={`space-y-4 ${className}`}>
        <div className="flex items-end space-x-2 h-32">
          {Array.from({ length: 12 }).map((_, index) => (
            <div key={index} className="flex-1 flex flex-col justify-end">
              <div 
                className="bg-gray-200 dark:bg-gray-700 animate-pulse rounded-t"
                style={{ height: `${Math.random() * 80 + 20}%` }}
              />
            </div>
          ))}
        </div>
        <div className="flex space-x-2">
          {Array.from({ length: 6 }).map((_, index) => (
            <Skeleton key={index} variant="text" width="3rem" />
          ))}
        </div>
      </div>
    );
  }

  // Bar chart
  return (
    <div className={`space-y-4 ${className}`}>
      <div className="flex items-end space-x-2 h-32">
        {Array.from({ length: 8 }).map((_, index) => (
          <div key={index} className="flex-1">
            <Skeleton 
              variant="rectangle" 
              height={`${Math.random() * 70 + 30}%`}
              className="w-full"
            />
          </div>
        ))}
      </div>
      <div className="flex space-x-2">
        {Array.from({ length: 4 }).map((_, index) => (
          <Skeleton key={index} variant="text" width="4rem" />
        ))}
      </div>
    </div>
  );
}

interface LoadingStateProps {
  isLoading: boolean;
  children: ReactNode;
  skeleton?: ReactNode;
}

export function LoadingState({ isLoading, children, skeleton }: LoadingStateProps) {
  if (isLoading) {
    return skeleton ? <>{skeleton}</> : <Skeleton className="w-full h-32" />;
  }
  
  return <>{children}</>;
} 