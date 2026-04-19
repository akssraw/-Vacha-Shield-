import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeft, Shield, ShieldAlert, ShieldCheck } from "lucide-react";
import { Link } from "react-router-dom";
import { io, type Socket } from "socket.io-client";

import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";

type ThreatLevel = "idle" | "safe" | "warning" | "alert";
type AnalysisResult = {
  synthetic_probability: number;
  human_probability: number;
  alert: boolean;
  threshold: number;
  latency_ms?: number;
  processing_ms?: number;
  decision_summary?: string;
  decision_mode?: string;
  verdict?: string;
  processing_state?: string;
  fraud_language_probability?: number;
  fraud_language_alert?: boolean;
  fraud_keywords?: string[];
  transcript_preview?: string;
};
type SpeechCtor = new () => BrowserSpeechRecognition;
type SpeechAlternativeLike = { transcript: string; confidence?: number };
type SpeechResultLike = {
  isFinal: boolean;
  length: number;
  [index: number]: SpeechAlternativeLike | undefined;
};
type SpeechEventLike = Event & {
  resultIndex: number;
  results: ArrayLike<SpeechResultLike>;
};

interface BrowserSpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  maxAlternatives?: number;
  onend: ((event: Event) => void) | null;
  onerror: ((event: Event & { error?: string }) => void) | null;
  onresult: ((event: SpeechEventLike) => void) | null;
  start(): void;
  stop(): void;
  abort(): void;
}

declare global {
  interface Window {
    SpeechRecognition?: SpeechCtor;
    webkitSpeechRecognition?: SpeechCtor;
  }
}

const AUDIO_THRESHOLD = 0.018;
const MAX_LINES = 120;
const KEYWORDS = [
  "otp",
  "kyc",
  "refund",
  "verification code",
  "bank account",
  "upi pin",
  "cvv",
  "card number",
  "remote access",
  "screen share",
  "urgent",
  "gift card",
];
const HIGH_RISK = new Set(["otp", "verification code", "upi pin", "cvv", "card number", "remote access", "screen share"]);

const THREAT_TEXT: Record<ThreatLevel, { title: string; detail: string }> = {
  idle: { title: "Ready", detail: "Start the monitor before the call and keep the caller on speaker." },
  safe: { title: "Listening", detail: "Vacha Shield is checking the live voice and scam language." },
  warning: { title: "Be careful", detail: "The call is showing suspicious signs. Slow down and verify." },
  alert: { title: "Likely fraud", detail: "The voice or scam wording looks dangerous. End the call and verify." },
};

const baseUrl = () => {
  const configured = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
  if (configured) return configured;
  const host = window.location.hostname;
  return host === "localhost" || host === "127.0.0.1" ? "http://127.0.0.1:5000" : "";
};

const normalize = (data: unknown): AnalysisResult => {
  const payload = typeof data === "object" && data !== null ? (data as Record<string, unknown>) : {};
  const synthetic = Number(payload.synthetic_probability ?? payload.fake_probability ?? payload.probability ?? 0);
  const threshold = Number(payload.threshold ?? 0.5);
  const human = Number(payload.human_probability ?? 1 - synthetic);
  return {
    synthetic_probability: Number.isFinite(synthetic) ? synthetic : 0,
    human_probability: Number.isFinite(human) ? human : 0,
    threshold: Number.isFinite(threshold) ? threshold : 0.5,
    alert: Boolean(payload.alert ?? synthetic > threshold),
    latency_ms: Number.isFinite(payload.latency_ms) ? Number(payload.latency_ms) : undefined,
    processing_ms: Number.isFinite(payload.processing_ms) ? Number(payload.processing_ms) : undefined,
    decision_summary: typeof payload.decision_summary === "string" ? payload.decision_summary : undefined,
    decision_mode: typeof payload.decision_mode === "string" ? payload.decision_mode : undefined,
    verdict: typeof payload.verdict === "string" ? payload.verdict : undefined,
    processing_state: typeof payload.processing_state === "string" ? payload.processing_state : undefined,
    fraud_language_probability: Number.isFinite(payload.fraud_language_probability) ? Number(payload.fraud_language_probability) : undefined,
    fraud_language_alert: Boolean(payload.fraud_language_alert),
    fraud_keywords: Array.isArray(payload.fraud_keywords) ? payload.fraud_keywords.filter((item): item is string => typeof item === "string") : undefined,
    transcript_preview: typeof payload.transcript_preview === "string" ? payload.transcript_preview : undefined,
  };
};

const countHits = (source: string, query: string) => (source.match(new RegExp(query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "g"))?.length ?? 0);
const since = (stamp: number | null) => (!stamp ? "Waiting" : Date.now() - stamp < 2000 ? "Just now" : `${Math.round((Date.now() - stamp) / 1000)}s ago`);

const Meter = ({ label, value, danger = false }: { label: string; value: number; danger?: boolean }) => (
  <div className="rounded-3xl border border-white/10 bg-black/10 p-5">
    <div className="mb-3 flex items-end justify-between gap-4">
      <p className="font-mono text-[10px] uppercase tracking-[0.28em] text-muted-foreground">{label}</p>
      <p className="font-display text-2xl font-bold">{Math.round(value * 100)}%</p>
    </div>
    <div className="h-3 overflow-hidden rounded-full bg-white/10">
      <div className={`h-full rounded-full ${danger ? "bg-gradient-to-r from-destructive to-[#ff9f7d]" : "bg-gradient-to-r from-primary to-[#c6ff7a]"}`} style={{ width: `${Math.max(0, Math.min(value * 100, 100))}%` }} />
    </div>
  </div>
);

const Stat = ({ label, value, hint }: { label: string; value: string; hint: string }) => (
  <div className="rounded-3xl border border-white/10 bg-black/10 p-5">
    <p className="font-mono text-[10px] uppercase tracking-[0.28em] text-muted-foreground">{label}</p>
    <p className="mt-2 font-display text-2xl font-bold">{value}</p>
    <p className="mt-2 text-xs text-muted-foreground">{hint}</p>
  </div>
);

export default function CallMonitorPage() {
  const { toast } = useToast();
  const [isStarting, setIsStarting] = useState(false);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [monitorError, setMonitorError] = useState<string | null>(null);
  const [analysisState, setAnalysisState] = useState("Waiting to start.");
  const [audioLevel, setAudioLevel] = useState(0);
  const [heardAudioAt, setHeardAudioAt] = useState<number | null>(null);
  const [lastAnalysisAt, setLastAnalysisAt] = useState<number | null>(null);
  const [voiceResult, setVoiceResult] = useState<AnalysisResult | null>(null);
  const [transcriptLines, setTranscriptLines] = useState<string[]>([]);
  const [interimTranscript, setInterimTranscript] = useState("");
  const [keywordCounts, setKeywordCounts] = useState<Record<string, number>>({});
  const [speechStatus, setSpeechStatus] = useState("Keyword scan waiting.");
  const [warningCount, setWarningCount] = useState(0);

  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const workletRef = useRef<AudioWorkletNode | null>(null);
  const socketRef = useRef<Socket | null>(null);
  const meterRef = useRef<number | null>(null);
  const recognitionRef = useRef<BrowserSpeechRecognition | null>(null);
  const recognitionRestartRef = useRef<number | null>(null);
  const transcriptScrollRef = useRef<HTMLDivElement | null>(null);
  const isMonitoringRef = useRef(false);
  const lastAlertRef = useRef(false);
  const syntheticHistoryRef = useRef<number[]>([]);
  const speechCtor = useMemo(() => (window.SpeechRecognition ?? window.webkitSpeechRecognition) as SpeechCtor | undefined, []);

  const keywordEntries = useMemo(() => Object.entries(keywordCounts).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])), [keywordCounts]);
  const keywordRisk = useMemo(() => {
    const total = keywordEntries.reduce((sum, [, count]) => sum + count, 0);
    const repeated = keywordEntries.filter(([, count]) => count >= 2);
    const maxRepeat = keywordEntries.reduce((max, [, count]) => Math.max(max, count), 0);
    const highRisk = keywordEntries.some(([keyword]) => HIGH_RISK.has(keyword));
    let score = 0;
    if (total >= 1) score += 0.2;
    if (total >= 3) score += 0.12;
    if (repeated.length >= 1) score += 0.2;
    if (repeated.length >= 2) score += 0.12;
    if (maxRepeat >= 3) score += 0.15;
    if (highRisk) score += 0.15;
    return Math.min(score, 0.99);
  }, [keywordEntries]);

  const isAudioDetected = !!heardAudioAt && Date.now() - heardAudioAt < 2500;
  const voiceRisk = voiceResult?.synthetic_probability ?? 0;
  const backendKeywordRisk = voiceResult?.fraud_language_probability ?? 0;
  const combinedKeywordRisk = Math.max(keywordRisk, backendKeywordRisk);
  const threat = useMemo<ThreatLevel>(() => {
    if (!isMonitoring) return "idle";
    if (voiceRisk >= 0.72 || combinedKeywordRisk >= 0.6 || Boolean(voiceResult?.alert)) return "alert";
    if (voiceRisk >= 0.42 || combinedKeywordRisk >= 0.35 || !!monitorError) return "warning";
    return "safe";
  }, [combinedKeywordRisk, isMonitoring, monitorError, voiceResult?.alert, voiceRisk]);

  const statusTitle = THREAT_TEXT[threat].title;
  const statusDetail = THREAT_TEXT[threat].detail;
  const transcriptPreview = useMemo(() => [transcriptLines.join(" "), interimTranscript].filter(Boolean).join(" "), [interimTranscript, transcriptLines]);
  const transcriptDisplayLines = useMemo(() => {
    const lines = voiceResult?.transcript_preview?.trim()
      ? [voiceResult.transcript_preview.trim()]
      : transcriptLines;
    return interimTranscript ? [...lines, interimTranscript] : lines;
  }, [interimTranscript, transcriptLines, voiceResult?.transcript_preview]);
  const verdict = useMemo(() => {
    if (!isMonitoring) return "Ready";
    if (threat === "alert" && combinedKeywordRisk >= voiceRisk) return "Likely fraud call";
    if (threat === "alert") return "Likely AI voice";
    if (voiceResult) return voiceRisk >= 0.5 ? "Possibly AI voice" : "Likely human voice";
    return isAudioDetected ? "Listening to speaker audio" : "Waiting for speaker audio";
  }, [combinedKeywordRisk, isAudioDetected, isMonitoring, threat, voiceResult, voiceRisk]);

  const stopSpeech = useCallback(() => {
    if (recognitionRestartRef.current != null) window.clearTimeout(recognitionRestartRef.current);
    recognitionRestartRef.current = null;
    const recognition = recognitionRef.current;
    recognitionRef.current = null;
    if (!recognition) return;
    recognition.onresult = null;
    recognition.onerror = null;
    recognition.onend = null;
    try { recognition.stop(); } catch { try { recognition.abort(); } catch { /* no-op */ } }
  }, []);

  const addKeywordHits = useCallback((phrase: string) => {
    const lowered = phrase.toLowerCase();
    const hits: Record<string, number> = {};
    for (const keyword of KEYWORDS) {
      const count = countHits(lowered, keyword);
      if (count > 0) hits[keyword] = count;
    }
    if (!Object.keys(hits).length) return;
    setKeywordCounts((current) => {
      const next = { ...current };
      for (const [keyword, count] of Object.entries(hits)) next[keyword] = (next[keyword] ?? 0) + count;
      return next;
    });
  }, []);

  const startSpeech = useCallback(() => {
    if (!speechCtor) {
      setSpeechStatus("Keyword scan is not supported in this browser.");
      return;
    }
    stopSpeech();
    const recognition = new speechCtor();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-IN";
    recognition.maxAlternatives = 3;
    recognition.onresult = (event) => {
      const finalPhrases: string[] = [];
      const finalKeywordPhrases: string[] = [];
      let interim = "";
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const result = event.results[index];
        if (!result) continue;
        const alternatives: string[] = [];
        const alternativeCount = Math.min(result.length || 1, 3);
        for (let altIndex = 0; altIndex < alternativeCount; altIndex += 1) {
          const transcript = result[altIndex]?.transcript?.trim();
          if (transcript) alternatives.push(transcript);
        }
        if (!alternatives.length) continue;
        if (result.isFinal) {
          finalPhrases.push(alternatives[0]);
          finalKeywordPhrases.push(alternatives.join(" "));
        } else {
          interim = alternatives[0];
        }
      }
      if (finalPhrases.length) {
        setTranscriptLines((current) => [...current, ...finalPhrases].slice(-MAX_LINES));
        finalKeywordPhrases.forEach(addKeywordHits);
      }
      setInterimTranscript(interim);
      setSpeechStatus("Keyword scan is listening.");
    };
    recognition.onerror = (event) => {
      const code = event.error ?? "speech-error";
      setSpeechStatus(code === "not-allowed" || code === "service-not-allowed" ? "Keyword scan was blocked by the browser." : "Keyword scan reconnecting...");
    };
    recognition.onend = () => {
      recognitionRef.current = null;
      setInterimTranscript("");
      if (!isMonitoringRef.current) return;
      recognitionRestartRef.current = window.setTimeout(() => {
        recognitionRestartRef.current = null;
        if (isMonitoringRef.current) startSpeech();
      }, 800);
    };
    try {
      recognition.start();
      recognitionRef.current = recognition;
      setSpeechStatus("Keyword scan is listening.");
    } catch {
      setSpeechStatus("Keyword scan could not start. Voice analysis still works.");
    }
  }, [addKeywordHits, speechCtor, stopSpeech]);

  const stopMeter = useCallback(() => {
    if (meterRef.current != null) window.cancelAnimationFrame(meterRef.current);
    meterRef.current = null;
  }, []);

  const cleanup = useCallback(() => {
    stopMeter();
    stopSpeech();
    socketRef.current?.emit("call_monitor_stop");
    socketRef.current?.disconnect();
    socketRef.current = null;
    workletRef.current?.disconnect();
    workletRef.current?.port.close();
    workletRef.current = null;
    sourceRef.current?.disconnect();
    sourceRef.current = null;
    analyserRef.current?.disconnect();
    analyserRef.current = null;
    const context = audioContextRef.current;
    audioContextRef.current = null;
    if (context && context.state !== "closed") void context.close();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    syntheticHistoryRef.current = [];
  }, [stopMeter, stopSpeech]);

  const applyLiveResult = useCallback((payload: unknown) => {
    const normalized = normalize(payload);
    const history = [...syntheticHistoryRef.current, normalized.synthetic_probability].slice(-5);
    syntheticHistoryRef.current = history;
    const smoothedRisk = history.reduce((sum, value) => sum + value, 0) / history.length;
    setVoiceResult({
      ...normalized,
      synthetic_probability: Number(smoothedRisk.toFixed(4)),
      human_probability: Number((1 - smoothedRisk).toFixed(4)),
      alert: normalized.alert || smoothedRisk > normalized.threshold,
    });
    setLastAnalysisAt(Date.now());
    setMonitorError(null);
    setAnalysisState("Live analysis running.");
  }, []);

  const startSocketEngine = useCallback(async (context: AudioContext, source: MediaStreamAudioSourceNode) => {
    await context.audioWorklet.addModule("/call-monitor-processor.js");
    const socket = io(baseUrl(), {
      transports: ["websocket", "polling"],
      reconnection: true,
      reconnectionDelay: 600,
      reconnectionDelayMax: 2500,
    });
    const worklet = new AudioWorkletNode(context, "call-monitor-processor");
    socketRef.current = socket;
    workletRef.current = worklet;

    socket.on("connect", () => {
      socket.emit("call_monitor_start", {
        sample_rate: Math.round(context.sampleRate),
        analysis: {
          analysis_profile: "strict",
          sensitivity: "0.74",
          decision_floor: "0.55",
          borderline_margin: "0.08",
        },
      });
      setAnalysisState("Monitor is live. Put the caller on speaker.");
    });
    socket.on("call_monitor_status", (payload: { message?: string }) => {
      if (payload?.message) setAnalysisState(payload.message);
    });
    socket.on("call_monitor_result", applyLiveResult);
    socket.on("call_monitor_error", (payload: { error?: string; detail?: string }) => {
      setMonitorError(payload?.detail || payload?.error || "Live analysis failed.");
    });
    socket.on("disconnect", () => {
      if (isMonitoringRef.current) setAnalysisState("Reconnecting live engine...");
    });
    worklet.port.onmessage = (event: MessageEvent<Float32Array>) => {
      const chunk = event.data;
      if (!isMonitoringRef.current || !socket.connected || !chunk?.length) return;
      socket.emit("call_monitor_chunk", {
        audio: chunk.buffer.slice(0),
        captured_at_ms: Date.now(),
      });
    };
    source.connect(worklet);
  }, [applyLiveResult]);

  const startMeter = useCallback((analyser: AnalyserNode) => {
    stopMeter();
    const buffer = new Uint8Array(analyser.fftSize);
    const tick = () => {
      analyser.getByteTimeDomainData(buffer);
      let sum = 0;
      for (const value of buffer) {
        const normalized = (value - 128) / 128;
        sum += normalized * normalized;
      }
      const level = Math.min(Math.sqrt(sum / buffer.length) * 4, 1);
      setAudioLevel(level);
      if (level >= AUDIO_THRESHOLD) setHeardAudioAt(Date.now());
      if (isMonitoringRef.current) meterRef.current = window.requestAnimationFrame(tick);
    };
    meterRef.current = window.requestAnimationFrame(tick);
  }, [stopMeter]);

  const startMonitor = useCallback(async () => {
    if (isStarting || isMonitoringRef.current) return;
    setIsStarting(true);
    setMonitorError(null);
    setAnalysisState("Starting monitor...");
    setVoiceResult(null);
    setLastAnalysisAt(null);
    setWarningCount(0);
    setKeywordCounts({});
    setTranscriptLines([]);
    setInterimTranscript("");
    setHeardAudioAt(null);
    syntheticHistoryRef.current = [];
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: true,
          channelCount: 1,
          sampleRate: 48000,
          sampleSize: 16,
        },
      });
      const context = new AudioContext({ latencyHint: "interactive" });
      await context.resume();
      const source = context.createMediaStreamSource(stream);
      const analyser = context.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      streamRef.current = stream;
      audioContextRef.current = context;
      sourceRef.current = source;
      analyserRef.current = analyser;
      isMonitoringRef.current = true;
      setIsMonitoring(true);
      setAnalysisState("Monitor is live. Put the caller on speaker.");
      startMeter(analyser);
      await startSocketEngine(context, source);
      startSpeech();
      toast({ title: "Monitor started", description: "The monitor will stay on until you press Stop Monitor." });
    } catch (error) {
      cleanup();
      isMonitoringRef.current = false;
      setIsMonitoring(false);
      setMonitorError(error instanceof Error ? error.message : "Could not start the monitor.");
      setAnalysisState("Monitor could not start.");
    } finally {
      setIsStarting(false);
    }
  }, [cleanup, isStarting, startMeter, startSocketEngine, startSpeech, toast]);

  const stopMonitor = useCallback(() => {
    isMonitoringRef.current = false;
    setIsMonitoring(false);
    setIsStarting(false);
    cleanup();
    setAnalysisState("Monitor stopped.");
    setMonitorError(null);
    setAudioLevel(0);
    setHeardAudioAt(null);
    setInterimTranscript("");
  }, [cleanup]);

  useEffect(() => () => {
    isMonitoringRef.current = false;
    cleanup();
  }, [cleanup]);

  useEffect(() => {
    const alertNow = isMonitoring && threat === "alert";
    if (alertNow && !lastAlertRef.current) {
      setWarningCount((current) => current + 1);
      if (typeof navigator !== "undefined" && typeof navigator.vibrate === "function") navigator.vibrate([140, 80, 200]);
    }
    lastAlertRef.current = alertNow;
  }, [isMonitoring, threat]);

  useEffect(() => {
    const box = transcriptScrollRef.current;
    if (box) box.scrollTop = box.scrollHeight;
  }, [transcriptDisplayLines]);

  return (
    <div className="relative min-h-screen overflow-hidden bg-background text-foreground">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_10%_0%,rgba(116,255,163,0.14),transparent_32%),radial-gradient(circle_at_90%_4%,rgba(255,107,107,0.12),transparent_24%)]" />
      <nav className="relative z-10 flex flex-col gap-4 border-b border-white/10 px-6 py-5 md:flex-row md:items-center md:justify-between md:px-8">
        <div className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-3xl bg-primary/12"><Shield className="h-5 w-5 text-primary" /></div>
          <div>
            <h1 className="font-display text-xl font-bold">Call Monitor</h1>
            <p className="font-mono text-[10px] uppercase tracking-[0.35em] text-muted-foreground">Live Scam Protection</p>
          </div>
        </div>
        <Button asChild variant="outline">
          <Link to="/"><ArrowLeft className="mr-2 h-4 w-4" />Back to Analysis</Link>
        </Button>
      </nav>
      <main className="relative z-10 mx-auto max-w-5xl px-6 py-10 md:px-8">
        <section className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          <div className="rounded-[32px] border border-white/10 bg-black/10 p-6">
            <h2 className="font-display text-4xl font-bold leading-tight">Start the monitor, answer the call, and put the caller on speaker.</h2>
            <p className="mt-4 text-base leading-relaxed text-muted-foreground">Vacha Shield keeps listening through your microphone, checks if the voice sounds AI or human, and flags scam wording like OTP, KYC, CVV, refund, or remote access.</p>
            <div className="mt-6 space-y-3 text-sm text-muted-foreground">
              <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3">1. Tap <span className="font-semibold text-foreground">Start Monitor</span>.</div>
              <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3">2. Accept the call and switch to speaker mode.</div>
              <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3">3. Keep this page open. It will run until you press Stop Monitor.</div>
            </div>
          </div>
          <div className={`rounded-[32px] border bg-black/10 p-6 ${threat === "alert" ? "border-destructive/35" : threat === "warning" ? "border-warning/30" : "border-white/10"}`}>
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="font-mono text-[10px] uppercase tracking-[0.28em] text-muted-foreground">Live status</p>
                <h3 className="mt-2 font-display text-3xl font-bold">{statusTitle}</h3>
              </div>
              <div className={`flex h-14 w-14 items-center justify-center rounded-3xl ${threat === "alert" ? "bg-destructive/15 text-destructive" : threat === "warning" ? "bg-warning/15 text-warning" : "bg-primary/12 text-primary"}`}>
                {threat === "alert" ? <ShieldAlert className="h-7 w-7" /> : threat === "idle" ? <Shield className="h-7 w-7" /> : <ShieldCheck className="h-7 w-7" />}
              </div>
            </div>
            <p className="mt-4 text-sm leading-relaxed text-muted-foreground">{statusDetail}</p>
            <div className="mt-5 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm">
              <p className="font-semibold text-foreground">{verdict}</p>
              <p className="mt-1 text-muted-foreground">{analysisState}</p>
            </div>
            {monitorError && <div className="mt-4 rounded-2xl border border-destructive/25 bg-destructive/10 px-4 py-3 text-sm text-destructive">{monitorError}</div>}
            <div className="mt-6 flex flex-wrap gap-3">
              <Button onClick={isMonitoring ? stopMonitor : startMonitor} disabled={isStarting} className={isMonitoring ? "bg-destructive hover:bg-destructive/85" : "bg-gradient-to-r from-primary to-[#c6ff7a] text-primary-foreground"}>
                {isStarting ? "Starting..." : isMonitoring ? "Stop Monitor" : "Start Monitor"}
              </Button>
            </div>
          </div>
        </section>

        <section className="mt-6 grid gap-4 md:grid-cols-3">
          <Stat label="Monitor" value={isMonitoring ? "Active" : "Off"} hint="It stays on until you press Stop Monitor." />
          <Stat label="Speaker Audio" value={isAudioDetected ? "Heard" : isMonitoring ? "Waiting" : "Off"} hint="The mic must hear the caller on speaker." />
          <Stat label="Warnings" value={String(warningCount).padStart(2, "0")} hint="How many times the call crossed the danger threshold." />
        </section>

        <section className="mt-6 grid gap-6 lg:grid-cols-[1fr_1fr]">
          <div className="space-y-4">
            <Meter label="AI Voice Risk" value={voiceRisk} danger />
            <Meter label="Human Voice Confidence" value={voiceResult?.human_probability ?? 0} />
            <Meter label="Scam Keyword Risk" value={combinedKeywordRisk} danger />
          </div>
          <div className="rounded-[32px] border border-white/10 bg-black/10 p-6">
            <h3 className="font-display text-2xl font-bold">Live reading</h3>
            <div className="mt-5 grid gap-4 md:grid-cols-2">
              <Stat label="Last Check" value={since(lastAnalysisAt)} hint="How recently the voice model finished a chunk." />
              <Stat label="Speed" value={voiceResult?.latency_ms != null || voiceResult?.processing_ms != null ? `${Math.round(voiceResult?.latency_ms ?? voiceResult?.processing_ms ?? 0)} ms` : "--"} hint="Approximate time for the last chunk." />
            </div>
            <div className="mt-5 rounded-3xl border border-white/10 bg-white/5 p-5">
              <p className="font-medium">Transcript preview</p>
              <div ref={transcriptScrollRef} className="mt-3 max-h-44 min-h-20 overflow-y-auto pr-3 text-sm leading-relaxed text-muted-foreground">
                {transcriptDisplayLines.length ? (
                  transcriptDisplayLines.map((line, index) => (
                    <p key={`${index}-${line.slice(0, 18)}`} className="mb-2 last:mb-0">{line}</p>
                  ))
                ) : (
                  <p>Transcript preview will appear here when speech is clear enough for transcription.</p>
                )}
              </div>
              <p className="mt-3 text-xs text-muted-foreground">{speechStatus}</p>
            </div>
            {voiceResult?.decision_summary && (
              <div className="mt-4 rounded-3xl border border-white/10 bg-white/5 p-5">
                <p className="font-medium">Why this result</p>
                <p className="mt-3 text-sm leading-relaxed text-muted-foreground">{voiceResult.decision_summary}</p>
              </div>
            )}
          </div>
        </section>

        <section className="mt-6 grid gap-6 lg:grid-cols-[1fr_0.9fr]">
          <div className="rounded-[32px] border border-white/10 bg-black/10 p-6">
            <h3 className="font-display text-2xl font-bold">Scam words spotted</h3>
            <div className="mt-5 flex flex-wrap gap-2">
              {(keywordEntries.length
                ? keywordEntries.slice(0, 6)
                : (voiceResult?.fraud_keywords?.length
                    ? voiceResult.fraud_keywords.slice(0, 6).map((keyword) => [keyword, 1] as const)
                    : KEYWORDS.slice(0, 6).map((keyword) => [keyword, 0] as const))
              ).map(([keyword, count]) => (
                <span key={keyword} className={`rounded-full border px-3 py-1 text-xs uppercase tracking-[0.16em] ${count > 0 ? "border-warning/40 bg-warning/10 text-warning" : "border-white/10 bg-white/5 text-muted-foreground"}`}>
                  {keyword}{count > 0 ? ` x${count}` : ""}
                </span>
              ))}
            </div>
          </div>
          <div className="rounded-[32px] border border-white/10 bg-black/10 p-6">
            <h3 className="font-display text-2xl font-bold">If the app warns</h3>
            <p className="mt-3 text-sm leading-relaxed text-muted-foreground">Stop sharing details, end the call, and verify the person through a number or app you opened yourself.</p>
            <div className="mt-5 rounded-2xl border border-white/10 bg-white/5 px-4 py-4">
              <p className="text-sm font-medium text-foreground">Live mic level</p>
              <p className="mt-2 text-3xl font-bold">{Math.round(audioLevel * 100)}%</p>
              <p className="mt-2 text-sm text-muted-foreground">Useful for checking whether the browser can hear the speakerphone audio.</p>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
