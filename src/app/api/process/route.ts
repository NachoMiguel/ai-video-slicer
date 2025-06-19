import { NextResponse } from 'next/server'

export async function POST(request: Request) {
  try {
    const formData = await request.formData()
    const sessionId = formData.get('session_id') as string
    const action = formData.get('action') as string

    if (!sessionId) {
      return NextResponse.json(
        { error: 'Session ID is required' },
        { status: 400 }
      )
    }

    if (!action) {
      return NextResponse.json(
        { error: 'Action is required' },
        { status: 400 }
      )
    }

    // Route to appropriate backend endpoint based on action
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000'
    
    let endpoint = ''
    switch (action) {
      case 'generate_full_script':
        endpoint = '/api/script/generate-full-script'
        break
      case 'modify_text':
        endpoint = '/api/script/modify-text'
        break
      case 'apply_modification':
        endpoint = '/api/script/apply-modification'
        break
      case 'modify_bulk_text':
        endpoint = '/api/script/modify-bulk-text'
        break
      case 'apply_bulk_modification':
        endpoint = '/api/script/apply-bulk-modification'
        break
      default:
        return NextResponse.json(
          { error: 'Invalid action' },
          { status: 400 }
        )
    }

    // Forward the request to the backend
    const response = await fetch(`${backendUrl}${endpoint}`, {
      method: 'POST',
      body: formData
    })

    if (!response.ok) {
      const errorData = await response.json()
      return NextResponse.json(
        { error: errorData.detail || 'Backend request failed' },
        { status: response.status }
      )
    }

    const data = await response.json()
    return NextResponse.json(data)

  } catch (error) {
    console.error('Error processing request:', error)
    return NextResponse.json(
      { error: 'Failed to process request' },
      { status: 500 }
    )
  }
} 