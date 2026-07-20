import type { AnalysisSettings } from '../types'

interface SettingsPanelProps {
  settings: AnalysisSettings
  backgroundRoiSet: boolean
  previewThreshold: boolean
  onChange: (settings: AnalysisSettings) => void
  onPreviewThresholdChange: (enabled: boolean) => void
}

export function SettingsPanel({
  settings,
  backgroundRoiSet,
  previewThreshold,
  onChange,
  onPreviewThresholdChange,
}: SettingsPanelProps) {
  const set = (patch: Partial<AnalysisSettings>) => onChange({ ...settings, ...patch })

  return (
    <section className="panel">
      <div className="panel-title">
        <span className="eyebrow">Analysis settings</span>
      </div>

      <div className="field">
        <label htmlFor="bg-percentile">Background percentile</label>
        <input
          id="bg-percentile"
          type="number"
          min={0}
          max={100}
          step={1}
          value={settings.background_percentile}
          onChange={(event) => set({ background_percentile: Number(event.target.value) })}
        />
      </div>

      <div className="field">
        <label htmlFor="kernel">Mask cleanup kernel</label>
        <input
          id="kernel"
          type="number"
          min={1}
          step={2}
          value={settings.morphological_kernel_size}
          onChange={(event) => set({ morphological_kernel_size: Number(event.target.value) })}
        />
      </div>

      <div className="field">
        <label htmlFor="noise-floor">Noise floor (L*)</label>
        <input
          id="noise-floor"
          type="number"
          min={0}
          max={100}
          step={0.5}
          value={settings.noise_floor_threshold}
          onChange={(event) => set({ noise_floor_threshold: Number(event.target.value) })}
        />
      </div>

      <div className="field">
        <label htmlFor="manual-threshold">Manual threshold (L*)</label>
        <input
          id="manual-threshold"
          type="number"
          min={0}
          max={100}
          step={0.5}
          value={settings.manual_threshold}
          disabled={backgroundRoiSet}
          onChange={(event) => set({ manual_threshold: Number(event.target.value) })}
        />
      </div>
      {backgroundRoiSet ? (
        <div className="field-hint">
          The background region sets the threshold; the manual value is ignored.
        </div>
      ) : (
        <div className="field-hint">
          Applied when no background region is marked. 0 disables thresholding.
        </div>
      )}

      <label className="switch-row">
        <input
          type="checkbox"
          checked={previewThreshold}
          disabled={backgroundRoiSet || settings.manual_threshold <= 0}
          onChange={(event) => onPreviewThresholdChange(event.target.checked)}
        />
        Preview threshold on the frame
      </label>
    </section>
  )
}
