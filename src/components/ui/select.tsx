"use client"

import * as React from "react"
import { ChevronDown } from "lucide-react"

export interface SelectOption {
  value: string | number
  label: string
  description?: string
}

export interface SelectProps {
  value: string | number
  onValueChange: (value: string | number) => void
  options: SelectOption[]
  placeholder?: string
  disabled?: boolean
  className?: string
  label?: string
}

const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
  ({ value, onValueChange, options, placeholder, disabled = false, className = "", label, ...props }, ref) => {
    const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
      const selectedValue = e.target.value
      // Try to convert to number if it's a numeric string
      const numericValue = Number(selectedValue)
      onValueChange(isNaN(numericValue) ? selectedValue : numericValue)
    }

    return (
      <div className={`space-y-2 ${className}`}>
        {label && (
          <label className="text-sm font-medium text-foreground">{label}</label>
        )}
        <div className="relative">
          <select
            value={value}
            onChange={handleChange}
            disabled={disabled}
            ref={ref}
            className={`
              w-full px-3 py-2 bg-background border border-border rounded-md 
              text-foreground appearance-none cursor-pointer focus:outline-none 
              focus:ring-2 focus:ring-primary focus:border-transparent
              disabled:cursor-not-allowed disabled:opacity-50
              pr-10
            `}
            {...props}
          >
            {placeholder && (
              <option value="" disabled>
                {placeholder}
              </option>
            )}
            {options.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          <ChevronDown className="absolute right-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
        </div>
        {/* Show description for selected option */}
        {options.find(opt => opt.value === value)?.description && (
          <p className="text-xs text-muted-foreground">
            {options.find(opt => opt.value === value)?.description}
          </p>
        )}
      </div>
    )
  }
)

Select.displayName = "Select"

export { Select } 