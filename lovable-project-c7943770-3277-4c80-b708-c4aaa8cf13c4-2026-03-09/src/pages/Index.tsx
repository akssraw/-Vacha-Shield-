import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  AlertTriangle,
  Download,
  Mic,
  MicOff,
  RotateCcw,
  Shield,
  ShieldCheck,
  Upload,
  Waves,
} from "lucide-react";
import { Link } from "react-router-dom";

import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";

type AnalysisResult = {
  synthetic_probability: number;
  human_probability: number;
  alert: boolean;
  threshold: number;
  spectrogram_base64?: string | null;
  model_probability?: number;
  artifact_probability?: number;
  analysis_profile?: string;
};

const AUDIO_EXTENSIONS = /\.(wav|mp3|webm)$/i;
const RECORDER_MIME_CANDIDATES = ["audio/webm;codecs=opus", "audio/webm"];

const ShieldLogo = () => (
  <div className="relative">
    <div
      className="absolute inset-0 blur-2xl opacity-40"
      style={{ background: "radial-gradient(circle, hsl(152 60% 42%) 0%, transparent 72%)" }}
    />
    <svg width="44" height="44" viewBox="0 0 24 24" fill="none" className="relative z-10">
      <defs>
        <linearGradient id="home-logo-grad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="hsl(152 60% 42%)" />
          <stop offset="100%" stopColor="hsl(164 70% 58%)" />
        </linearGradient>
      </defs>
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" stroke="url(#home-logo-grad)" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" stroke="url(#home-logo-grad)" strokeWidth="1.5" strokeLinecap="round" />
      <line x1="12" y1="19" x2="12" y2="23" stroke="url(#home-logo-grad)" strokeWidth="1.5" strokeLinecap="round" />
      <line x1="8" y1="23" x2="16" y2="23" stroke="url(#home-logo-grad)" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  </div>
);

const selectRecorderMimeType = () => {
  if (typeof MediaRecorder === "undefined") {
    throw new Error("MediaRecorder is not available in this browser.");
  }

  const supported = RECORDER_MIME_CANDIDATES.find((candidate) => {
    if (typeof MediaRecorder.isTypeSupported !== "function") {
      return candidate === "audio/webm";
    }
    return MediaRecorder.isTypeSupported(candidate);
  });

  if (!supported) {
    throw new Error("This browser does not support live audio capture in a backend-compatible format.");
  }

  return supported;
};

const normalizeAnalysisResult = (data: any): AnalysisResult => {
  const synthetic = Number(data.synthetic_probability ?? data.fake_probability ?? data.probability ?? 0);
  const threshold = Number(data.threshold ?? 0.5);
  const human = Number(data.human_probability ?? 1 - synthetic);

  return {
    synthetic_probability: Number.isFinite(synthetic) ? synthetic : 0,
    human_probability: Number.isFinite(human) ? human : 0,
    alert: Boolean(data.alert ?? synthetic > threshold),
    threshold: Number.isFinite(threshold) ? threshold : 0.5,
    spectrogram_base64: data.spectrogram_base64 ?? null,
    model_probability: Number.isFinite(data?.model_probability) ? Number(data.model_probability) : undefined,
    artifact_probability: Number.isFinite(data?.artifact_probability) ? Number(data.artifact_probability) : undefined,
    analysis_profile: typeof data?.analysis_profile === "string" ? data.analysis_profile : undefined,
  };
};

const StatTile = ({ label, value, hint }: { label: string; value: string; hint: string }) => (
  <div className="rounded-3xl border border-white/10 bg-black/10 p-4">
    <p className="font-mono text-[10px] uppercase tracking-[0.28em] text-muted-foreground">{label}</p>
    <p className="mt-2 font-display text-2xl font-bold">{value}</p>
    <p className="mt-1 text-xs text-muted-foreground">{hint}</p>
  </div>
);

const ScoreRail = ({
  label,
  value,
  accentClass,
}: {
  label: string;
  value: number;
  accentClass: string;
}) => (
  <div>
    <div className="mb-3 flex items-end justify-between gap-4">
      <p className="font-mono text-[10px] uppercase tracking-[0.28em] text-muted-foreground">{label}</p>
      <span className={`font-mono text-2xl font-bold ${accentClass}`}>{(value * 100).toFixed(1)}%</span>
    </div>
    <div className="h-3 overflow-hidden rounded-full bg-muted">
      <motion.div
        className={`h-full rounded-full ${accentClass === "text-destructive" ? "bg-gradient-to-r from-destructive to-[#ff9c7e]" : "bg-gradient-to-r from-safe to-[#b6ffd1]"} progress-shimmer`}
        initial={{ width: 0 }}
        animate={{ width: `${Math.max(0, Math.min(value * 100, 100))}%` }}
        transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
      />
    </div>
  </div>
);

const Index = () => {
  const { toast } = useToast();

  const [currentFile, setCurrentFile] = useState<File | null>(null);
  const [currentFileOrigin, setCurrentFileOrigin] = useState<"upload" | "capture" | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [showFeedbackThanks, setShowFeedbackThanks] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const backendUrl = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
  const resolveBackendUrl = useCallback(() => {
    if (backendUrl) {
      return backendUrl;
    }

    const host = window.location.hostname;
    const isLocalHost = host === "localhost" || host === "127.0.0.1";
    if (isLocalHost && window.location.port !== "5000") {
      return "http://127.0.0.1:5000";
    }

    return "";
  }, [backendUrl]);

  const handleFileSelect = useCallback(
    (file: File, origin: "upload" | "capture") => {
      if (!AUDIO_EXTENSIONS.test(file.name)) {
        toast({
          title: "Invalid format",
          description: "Upload or capture a .wav, .mp3, or .webm audio file.",
          variant: "destructive",
        });
        return;
      }

      setCurrentFile(file);
      setCurrentFileOrigin(origin);
      setResult(null);
      setShowFeedbackThanks(false);
    },
    [toast]
  );

  const stopRecording = useCallback(() => {
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
    }
    mediaRecorderRef.current = null;
  }, []);

  const startRecording = useCallback(async () => {
    try {
      const mimeType = selectRecorderMimeType();
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });

      streamRef.current = stream;
      const recorder = new MediaRecorder(stream, { mimeType, audioBitsPerSecond: 64000 });
      audioChunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: recorder.mimeType || mimeType });
        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = null;

        const file = new File([blob], "quick_sample.webm", {
          type: recorder.mimeType || mimeType,
        });
        handleFileSelect(file, "capture");
        setIsRecording(false);
      };

      mediaRecorderRef.current = recorder;
      recorder.start(250);
      setIsRecording(true);
    } catch (error) {
      toast({
        title: "Microphone blocked",
        description: error instanceof Error ? error.message : "Grant microphone permissions to continue.",
        variant: "destructive",
      });
      setIsRecording(false);
    }
  }, [handleFileSelect, toast]);

  const analyzeFile = useCallback(async () => {
    if (!currentFile) {
      return;
    }

    setIsLoading(true);
    setResult(null);
    setShowFeedbackThanks(false);

    try {
      const formData = new FormData();
      formData.append("file", currentFile);
      formData.append("analysis_profile", "strict");

      const response = await fetch(`${resolveBackendUrl()}/detect_voice`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      setResult(normalizeAnalysisResult(data));
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (error) {
      toast({
        title: "Analysis failed",
        description: error instanceof Error ? error.message : "Could not reach the backend server.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  }, [currentFile, resolveBackendUrl, toast]);

  const submitFeedback = useCallback(
    async (label: "human" | "ai") => {
      if (!currentFile) {
        return;
      }

      try {
        const formData = new FormData();
        formData.append("file", currentFile);
        formData.append("label", label);

        const response = await fetch(`${resolveBackendUrl()}/feedback`, {
          method: "POST",
          body: formData,
        });
        if (!response.ok) {
          throw new Error(`Feedback failed (${response.status})`);
        }

        setShowFeedbackThanks(true);
        toast({
          title: "Feedback saved",
          description: "This clip was added to the continuous learning queue.",
        });
      } catch (error) {
        toast({
          title: "Feedback failed",
          description: error instanceof Error ? error.message : "Could not submit feedback to backend.",
          variant: "destructive",
        });
      }
    },
    [currentFile, resolveBackendUrl, toast]
  );

  const resetAnalysis = useCallback(() => {
    setCurrentFile(null);
    setCurrentFileOrigin(null);
    setResult(null);
    setShowFeedbackThanks(false);
  }, []);

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      stopRecording();
    };
  }, [stopRecording]);

  return (
    <div className="relative min-h-screen overflow-hidden bg-background text-foreground noise grid-overlay">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_12%_0%,rgba(106,197,134,0.18),transparent_32%),radial-gradient(circle_at_88%_10%,rgba(238,248,170,0.12),transparent_28%)]" />

      <nav className="relative z-20 flex flex-col gap-4 border-b border-border/40 px-6 py-5 md:flex-row md:items-center md:justify-between md:px-8">
        <div className="flex items-center gap-3">
          <ShieldLogo />
          <div>
            <h1 className="font-display text-xl font-bold">Vacha Shield</h1>
            <p className="font-mono text-[10px] uppercase tracking-[0.35em] text-muted-foreground">Deepfake Voice Defense</p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Link
            to="/call-monitor"
            className="rounded-full border border-primary/20 bg-primary/10 px-4 py-2 font-mono text-[11px] uppercase tracking-[0.28em] text-primary transition-colors hover:bg-primary/15"
          >
            Call Monitor
          </Link>
        </div>
      </nav>

      <main className="relative z-10 mx-auto max-w-6xl px-6 py-10 md:px-8">
        {result && (
          <motion.section
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8 glass rounded-[34px] p-6 shadow-[0_28px_90px_rgba(0,0,0,0.2)] md:p-8"
          >
            <div
              className={`rounded-[30px] border p-6 ${result.alert ? "border-destructive/40 bg-destructive/8" : "border-safe/30 bg-safe/8"}`}
              style={{ boxShadow: result.alert ? "var(--glow-danger)" : "var(--glow-safe)" }}
            >
              <div className="flex flex-col gap-6 xl:flex-row xl:items-start xl:justify-between">
                <div className="max-w-2xl">
                  <p className="font-mono text-[11px] uppercase tracking-[0.28em] text-muted-foreground">Latest result</p>
                  <h2 className={`mt-3 font-display text-4xl font-extrabold ${result.alert ? "text-destructive" : "text-safe"}`}>
                    {result.alert ? "Synthetic voice indicators detected" : "Voice appears human"}
                  </h2>
                  <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
                    {result.alert
                      ? "The clip crossed the active threshold and should be treated as suspicious."
                      : "The clip stayed below the alert threshold in the current strict analysis profile."}
                  </p>
                </div>
                <div className="grid gap-4 sm:grid-cols-2 xl:min-w-[22rem]">
                  <StatTile label="Threshold" value={`${Math.round(result.threshold * 100)}%`} hint="Decision boundary for this scan." />
                  <StatTile label="Profile" value={result.analysis_profile?.toUpperCase() ?? "STRICT"} hint="Backend scoring profile used." />
                </div>
              </div>

              <div className="mt-8 grid gap-8 xl:grid-cols-[0.95fr_1.05fr]">
                <div className="space-y-6">
                  <ScoreRail label="Human Probability" value={result.human_probability} accentClass="text-safe" />
                  <ScoreRail label="Synthetic Probability" value={result.synthetic_probability} accentClass="text-destructive" />

                  <div className="grid gap-4 sm:grid-cols-2">
                    <StatTile label="Model" value={result.model_probability != null ? `${Math.round(result.model_probability * 100)}%` : "--"} hint="CNN-only signal." />
                    <StatTile label="Artifacts" value={result.artifact_probability != null ? `${Math.round(result.artifact_probability * 100)}%` : "--"} hint="Acoustic spoof cues." />
                  </div>
                </div>

                <div>
                  <p className="font-mono text-[10px] uppercase tracking-[0.28em] text-muted-foreground">Acoustic map</p>
                  <div className="relative mt-3 h-[250px] overflow-hidden rounded-[28px] border border-white/10 bg-black/15">
                    {result.spectrogram_base64 ? (
                      <img src={result.spectrogram_base64} alt="Analysis spectrogram" className="h-full w-full object-cover" />
                    ) : (
                      <div className="flex h-full flex-col items-center justify-center text-center">
                        <Waves className="h-10 w-10 text-white/20" />
                        <p className="mt-3 font-mono text-[11px] uppercase tracking-[0.28em] text-white/50">No acoustic map returned</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              <div className="mt-8 flex flex-col gap-3 sm:flex-row">
                <Button onClick={resetAnalysis} variant="outline" className="font-mono text-[11px] uppercase tracking-[0.28em]">
                  <RotateCcw className="mr-2 h-3.5 w-3.5" />
                  Scan Another File
                </Button>
              </div>

              <div className="mt-8 rounded-[26px] border border-primary/20 bg-primary/[0.04] p-5 text-center">
                <h3 className="font-display text-xl font-bold">Continuous learning</h3>
                <p className="mt-2 text-sm text-muted-foreground">Label this result if you want it fed back into the training queue.</p>
                {!showFeedbackThanks ? (
                  <div className="mt-4 flex flex-wrap justify-center gap-3">
                    <Button size="sm" variant="outline" className="border-safe/40 text-safe hover:bg-safe/10" onClick={() => submitFeedback("human")}>
                      <ShieldCheck className="mr-1.5 h-3.5 w-3.5" />
                      Human
                    </Button>
                    <Button size="sm" variant="outline" className="border-destructive/40 text-destructive hover:bg-destructive/10" onClick={() => submitFeedback("ai")}>
                      <AlertTriangle className="mr-1.5 h-3.5 w-3.5" />
                      AI Clone
                    </Button>
                  </div>
                ) : (
                  <p className="mt-4 font-mono text-sm font-bold uppercase tracking-[0.28em] text-primary">Feedback queued</p>
                )}
              </div>
            </div>
          </motion.section>
        )}

        <motion.section
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 grid gap-8 lg:grid-cols-[1.05fr_0.95fr]"
        >
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/10 px-4 py-2 font-mono text-[11px] uppercase tracking-[0.28em] text-primary">
              <Shield className="h-3.5 w-3.5" />
              Instant voice scan
            </div>
            <h2 className="mt-5 max-w-xl font-display text-5xl font-extrabold leading-[0.98]">Upload a clip. Get the answer fast.</h2>
            <p className="mt-4 max-w-xl text-base leading-relaxed text-muted-foreground">
              This page is just for analysis. If you want live protection during a call, open the dedicated Call Monitor.
            </p>
          </div>

          <div className="glass rounded-[30px] p-5 shadow-[0_24px_80px_rgba(0,0,0,0.18)]">
            <p className="font-mono text-[10px] uppercase tracking-[0.28em] text-muted-foreground">Need ongoing protection?</p>
            <h3 className="mt-3 font-display text-3xl font-bold">Use the separate monitor page</h3>
            <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
              The live guardian now lives on its own page so this analysis screen stays short and focused.
            </p>
            <Button asChild className="mt-5 bg-gradient-to-r from-primary to-[#c2ff7a] text-primary-foreground">
              <Link to="/call-monitor">Open Call Monitor</Link>
            </Button>
          </div>
        </motion.section>

        <section className="glass rounded-[34px] p-6 shadow-[0_28px_90px_rgba(0,0,0,0.18)] md:p-8">
          <div className="grid gap-5 lg:grid-cols-2">
            <div
              className={`group relative min-h-[260px] cursor-pointer rounded-[28px] border-2 border-dashed p-8 text-center transition-all duration-300 ${isDragOver ? "border-primary bg-primary/6 scale-[1.01]" : "border-border hover:border-primary/40 hover:bg-primary/[0.03]"}`}
              onDragOver={(event) => {
                event.preventDefault();
                setIsDragOver(true);
              }}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={(event) => {
                event.preventDefault();
                setIsDragOver(false);
                if (event.dataTransfer.files.length) {
                  handleFileSelect(event.dataTransfer.files[0], "upload");
                }
              }}
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-3xl bg-primary/10 transition-transform group-hover:scale-110">
                <Upload className="h-7 w-7 text-primary" />
              </div>
              <h3 className="font-display text-2xl font-bold">Upload audio</h3>
              <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
                Drag in a voice note, voicemail, or recorded meeting clip.
              </p>
              <input
                ref={fileInputRef}
                type="file"
                accept=".wav,.mp3,.webm"
                className="hidden"
                onChange={(event) => event.target.files?.[0] && handleFileSelect(event.target.files[0], "upload")}
              />
            </div>

            <div className="relative min-h-[260px] rounded-[28px] border border-white/10 bg-black/10 p-8 text-center">
              <div className={`mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-3xl transition-all ${isRecording ? "bg-destructive/20 scale-110" : "bg-primary/10"}`}>
                {isRecording ? <MicOff className="h-7 w-7 text-destructive" /> : <Mic className="h-7 w-7 text-primary" />}
              </div>
              <h3 className="font-display text-2xl font-bold">Quick record</h3>
              <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
                Capture a short sample from your microphone and send it straight to analysis.
              </p>
              <Button
                onClick={isRecording ? stopRecording : startRecording}
                className={`mt-8 font-mono text-[11px] uppercase tracking-[0.28em] ${isRecording ? "bg-destructive hover:bg-destructive/85" : "bg-gradient-to-r from-primary to-[#c2ff7a] text-primary-foreground"}`}
              >
                {isRecording ? "Stop Recording" : "Start Recording"}
              </Button>
            </div>
          </div>

          {currentFile && (
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-6 rounded-[26px] border border-white/10 bg-black/10 p-5"
            >
              <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                <div className="flex items-center gap-3">
                  <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-primary/10">
                    <Waves className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <p className="font-mono text-sm font-medium text-primary">{currentFile.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {(currentFile.size / 1024).toFixed(1)} KB - {currentFileOrigin === "capture" ? "Microphone capture ready" : "Upload ready"}
                    </p>
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-3">
                  {currentFileOrigin === "capture" && (
                    <Button
                      variant="outline"
                      className="font-mono text-[11px] uppercase tracking-[0.28em]"
                      onClick={() => {
                        const url = URL.createObjectURL(currentFile);
                        const anchor = document.createElement("a");
                        anchor.href = url;
                        anchor.download = currentFile.name || "vacha_capture.webm";
                        anchor.click();
                        URL.revokeObjectURL(url);
                      }}
                    >
                      <Download className="mr-2 h-3.5 w-3.5" />
                      Download
                    </Button>
                  )}
                  <Button
                    onClick={analyzeFile}
                    disabled={isLoading}
                    className="bg-gradient-to-r from-primary to-[#c2ff7a] font-mono text-[11px] uppercase tracking-[0.28em] text-primary-foreground"
                  >
                    {isLoading ? "Analyzing..." : "Analyze"}
                  </Button>
                </div>
              </div>
            </motion.div>
          )}
        </section>

        <AnimatePresence mode="wait">
          {isLoading && (
            <motion.section
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="mt-8 glass rounded-[30px] p-12 text-center"
            >
              <div className="mx-auto mb-5 h-16 w-16 rounded-full border-2 border-primary/20 border-t-primary" style={{ animation: "spin-slow 1s linear infinite" }} />
              <h3 className="font-display text-2xl font-bold">Running analysis</h3>
              <p className="mt-2 font-mono text-[11px] uppercase tracking-[0.28em] text-muted-foreground">
                Extracting spectrograms and scoring the clip
              </p>
            </motion.section>
          )}
        </AnimatePresence>
      </main>

      <footer className="relative z-10 px-6 py-8 text-center font-mono text-[11px] uppercase tracking-[0.28em] text-muted-foreground/60 md:px-8">
        Vacha Shield - faster analysis, cleaner flow
      </footer>
    </div>
  );
};

export default Index;
