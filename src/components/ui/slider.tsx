"use client"

import * as React from "react"

export interface SliderProps {
  value: number
  onValueChange: (value: number) => void
  min?: number
  max?: number
  step?: number
  disabled?: boolean
  className?: string
  label?: string
  unit?: string
}

const Slider = React.forwardRef<HTMLInputElement, SliderProps>(
  ({ value, onValueChange, min = 0, max = 100, step = 1, disabled = false, className = "", label, unit, ...props }, ref) => {
    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      onValueChange(Number(e.target.value))
    }

    return (
      <div className={`space-y-2 ${className}`}>
        {label && (
          <div className="flex items-center justify-between text-sm">
            <label className="font-medium text-foreground">{label}</label>
            <span className="text-muted-foreground">
              {value}{unit && ` ${unit}`}
            </span>
          </div>
        )}
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={handleChange}
          disabled={disabled}
          ref={ref}
          className={`
            w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer 
            dark:bg-gray-700 disabled:cursor-not-allowed disabled:opacity-50
            [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-4 
            [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:rounded-full 
            [&::-webkit-slider-thumb]:bg-primary [&::-webkit-slider-thumb]:cursor-pointer
            [&::-webkit-slider-thumb]:shadow-md hover:[&::-webkit-slider-thumb]:shadow-lg
            [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:rounded-full 
            [&::-moz-range-thumb]:bg-primary [&::-moz-range-thumb]:cursor-pointer 
            [&::-moz-range-thumb]:border-none [&::-moz-range-thumb]:shadow-md
          `}
          {...props}
        />
      </div>
    )
  }
)

Slider.displayName = "Slider"

export { Slider } 