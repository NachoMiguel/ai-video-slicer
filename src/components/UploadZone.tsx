import { useDropzone } from 'react-dropzone'
import { cn } from '../lib/utils'
import { useState } from 'react'

interface UploadZoneProps {
  onFilesAccepted: (files: File[]) => void
  maxFiles?: number
}

export function UploadZone({ onFilesAccepted, maxFiles = 2 }: UploadZoneProps) {
  const [error, setError] = useState<string | null>(null)

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'video/mp4': ['.mp4']
    },
    maxFiles,
    onDrop: (acceptedFiles) => {
      // Validate video duration
      const validateVideos = async () => {
        try {
          for (const file of acceptedFiles) {
            const duration = await getVideoDuration(file)
            if (duration > 1800) { // 30 minutes in seconds
              setError('Each video must be 30 minutes or less')
              return
            }
          }
          setError(null)
          onFilesAccepted(acceptedFiles)
        } catch (err) {
          setError('Error validating video duration')
        }
      }
      validateVideos()
    },
    onDropRejected: () => {
      setError('Please upload only MP4 files')
    }
  })

  const getVideoDuration = (file: File): Promise<number> => {
    return new Promise((resolve, reject) => {
      const video = document.createElement('video')
      video.preload = 'metadata'
      video.onloadedmetadata = () => {
        window.URL.revokeObjectURL(video.src)
        resolve(video.duration)
      }
      video.onerror = () => {
        reject('Error loading video')
      }
      video.src = URL.createObjectURL(file)
    })
  }

  return (
    <div className="w-full space-y-6">
      <div
        {...getRootProps()}
        className={cn(
          "border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-300",
          "hover:border-blue-400 hover:bg-blue-50/50 hover:scale-[1.02]",
          isDragActive ? "border-blue-500 bg-blue-50 scale-[1.02] shadow-lg" : "border-slate-300"
        )}
      >
        <input {...getInputProps()} />
        <div className="space-y-4">
          <div className="text-4xl">[CAM]</div>
          <div className="space-y-2">
            <p className="text-xl font-semibold text-slate-800">
              {isDragActive
                ? "Drop your videos here"
                : "Drag & drop videos here, or click to select"}
            </p>
            <p className="text-slate-600">
              Upload {maxFiles} videos (MP4 format, 30 minutes max each)
            </p>
          </div>
        </div>
      </div>

      {error && (
        <div className="rounded-xl bg-red-50 border border-red-200 p-4">
          <div className="flex items-center">
            <div className="text-red-500 mr-3">[WARNING]</div>
            <p className="text-sm font-medium text-red-800">{error}</p>
          </div>
        </div>
      )}
    </div>
  )
} 