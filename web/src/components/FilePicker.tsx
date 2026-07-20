import { useEffect, useState } from 'react'
import * as api from '../api'
import type { FsListing } from '../types'

interface FilePickerProps {
  error: string | null
  onPick: (path: string) => void
  onClose: () => void
}

export function FilePicker({ error, onPick, onClose }: FilePickerProps) {
  const [listing, setListing] = useState<FsListing | null>(null)
  const [pathInput, setPathInput] = useState('')
  const [browseError, setBrowseError] = useState<string | null>(null)

  const browse = async (path?: string, options?: { keepTypedPath?: boolean }) => {
    setBrowseError(null)
    try {
      const result = await api.listDirectory(path)
      setListing(result)
      if (options?.keepTypedPath) {
        // The initial listing must never clobber a path the user is typing.
        setPathInput((current) => (current === '' ? result.path : current))
      } else {
        setPathInput(result.path)
      }
    } catch (browseFailure) {
      setBrowseError(browseFailure instanceof Error ? browseFailure.message : String(browseFailure))
    }
  }

  useEffect(() => {
    void browse(undefined, { keepTypedPath: true })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose])

  const submitPath = () => {
    const trimmed = pathInput.trim()
    if (!trimmed) return
    // A file path opens directly; a directory browses into it.
    if (/\.(mp4|mov|avi|mkv|m4v|mpg|mpeg|wmv)$/i.test(trimmed)) {
      onPick(trimmed)
    } else {
      void browse(trimmed)
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(event) => event.stopPropagation()}>
        <div className="modal-head">
          <span className="eyebrow">Open video</span>
          <div className="modal-path">
            <input
              value={pathInput}
              placeholder="Paste a video path or browse below"
              onChange={(event) => setPathInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') submitPath()
              }}
            />
            <button className="btn" onClick={submitPath}>
              Go
            </button>
          </div>
        </div>
        <div className="modal-list">
          {listing?.parent && (
            <button className="fs-entry" onClick={() => void browse(listing.parent!)}>
              <span className="icon">↰</span> ..
            </button>
          )}
          {listing?.dirs.map((dir) => (
            <button key={dir} className="fs-entry" onClick={() => void browse(`${listing.path}/${dir}`)}>
              <span className="icon">▸</span> {dir}
            </button>
          ))}
          {listing?.videos.map((videoFile) => (
            <button
              key={videoFile}
              className="fs-entry video"
              onClick={() => onPick(`${listing.path}/${videoFile}`)}
            >
              <span className="icon">◈</span> {videoFile}
            </button>
          ))}
          {listing && listing.dirs.length === 0 && listing.videos.length === 0 && (
            <div style={{ color: 'var(--faint)', padding: 10 }}>No folders or videos here.</div>
          )}
        </div>
        {(error || browseError) && (
          <div className="error-text" style={{ padding: '0 16px' }}>
            {error || browseError}
          </div>
        )}
        <div className="modal-foot">
          <button className="btn" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  )
}
