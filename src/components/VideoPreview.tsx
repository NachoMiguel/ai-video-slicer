interface VideoPreviewProps {
  videoUrl: string;
  onDownload: () => void;
}

export function VideoPreview({ videoUrl, onDownload }: VideoPreviewProps) {
  return (
    <div className="panel h-full flex flex-col">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold text-gray-900">Video Preview</h2>
        <button
          onClick={onDownload}
          className="btn-primary flex items-center gap-2"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
          Download Video
        </button>
      </div>
      <div className="flex-1 relative">
        <video
          src={videoUrl}
          controls
          className="w-full h-full object-contain bg-black rounded-lg"
        />
      </div>
    </div>
  )
} 