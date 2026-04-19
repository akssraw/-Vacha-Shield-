import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeft, Bot, Play, Save, Shield, Sparkles, History, MessageSquareText } from "lucide-react";
import { Link } from "react-router-dom";

import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";

type Speaker = "Agent" | "Customer";
type SentimentTone = "Positive" | "Neutral" | "Negative";
type DashboardView = "History" | "NewAnalysis" | "LiveMonitor" | "ReportDetail";

interface CallReport {
  id: string;
  transcript: string;
  overallScore: number;
  sentiment: {
    initial: SentimentTone;
    final: SentimentTone;
  };
  scorecard: {
    greetingPassed: boolean;
    deEscalationPassed: boolean;
  };
  actionItems: string[];
}

interface LiveMessage {
  speaker: Speaker;
  text: string;
  timestamp: string;
}

const MOCK_AGENT_MESSAGES = [
  "Thanks for calling support, I’m here to help.",
  "I understand why that feels frustrating.",
  "Let me check the account notes for you.",
  "I can help sort that out right now.",
  "Thanks for staying with me while I verify the details.",
];

const MOCK_CUSTOMER_MESSAGES = [
  "I’ve already explained this twice and nothing changed.",
  "Why is my payment still showing as pending?",
  "I just need someone to fix this today.",
  "This is getting really frustrating for me.",
  "Can you please tell me what happens next?",
];

const WAIT_STEPS = [
  { label: "Parsing transcript...", delay: 700 },
  { label: "Analyzing sentiment...", delay: 900 },
  { label: "Generating scorecard...", delay: 700 },
  { label: "Preparing recommendations...", delay: 700 },
] as const;

const HISTORY_SEED: CallReport[] = [
  {
    id: "CALL-1042",
    transcript:
      "Agent: Thank you for calling support. Customer: My refund has not arrived yet. Agent: I understand the frustration and I will check it now.",
    overallScore: 87,
    sentiment: { initial: "Negative", final: "Neutral" },
    scorecard: { greetingPassed: true, deEscalationPassed: true },
    actionItems: ["Follow up with refund timeline in writing.", "Share a clear next-step summary before ending the call."],
  },
  {
    id: "CALL-1037",
    transcript:
      "Customer: I have called three times already. Agent: Let me investigate the issue. Customer: I need faster help on this ticket.",
    overallScore: 71,
    sentiment: { initial: "Negative", final: "Negative" },
    scorecard: { greetingPassed: false, deEscalationPassed: false },
    actionItems: ["Open with a clear greeting and ownership statement.", "Use empathy earlier when the customer shows frustration."],
  },
];

const wait = (ms: number) => new Promise<void>((resolve) => window.setTimeout(resolve, ms));

const formatTimestamp = (date = new Date()) =>
  date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });

const clampScore = (value: number) => Math.max(0, Math.min(100, Math.round(value)));

const inferSentiment = (text: string): SentimentTone => {
  const normalized = text.toLowerCase();
  const negativeSignals = ["angry", "frustrated", "upset", "issue", "problem", "pending", "escalate", "annoyed"];
  const positiveSignals = ["thanks", "great", "helpful", "resolved", "appreciate", "glad"];

  const negativeHits = negativeSignals.filter((term) => normalized.includes(term)).length;
  const positiveHits = positiveSignals.filter((term) => normalized.includes(term)).length;

  if (negativeHits > positiveHits) {
    return "Negative";
  }
  if (positiveHits > negativeHits) {
    return "Positive";
  }
  return "Neutral";
};

const buildActionItems = (greetingPassed: boolean, deEscalationPassed: boolean, finalSentiment: SentimentTone) => {
  const items: string[] = [];

  if (!greetingPassed) {
    items.push("Open with a clearer greeting and identify the next step early.");
  }
  if (!deEscalationPassed) {
    items.push("Acknowledge the customer emotion sooner and confirm ownership.");
  }
  if (finalSentiment === "Negative") {
    items.push("End with a stronger recap and concrete follow-up commitment.");
  }
  if (!items.length) {
    items.push("Keep this structure: clear greeting, empathy, recap, and next-step confirmation.");
  }

  return items;
};

const createCallReport = (id: string, transcript: string): CallReport => {
  const normalized = transcript.toLowerCase();
  const greetingPassed = /(hello|hi|good morning|good afternoon|thank you for calling|welcome)/.test(normalized);
  const deEscalationPassed = /(sorry|understand|i can help|let me help|i know this is frustrating|appreciate your patience)/.test(normalized);
  const initialSentiment = inferSentiment(transcript.slice(0, Math.max(80, Math.floor(transcript.length * 0.35))));
  const finalSentiment = inferSentiment(transcript.slice(Math.max(0, transcript.length - 160)));

  const baseScore =
    62 +
    (greetingPassed ? 14 : -6) +
    (deEscalationPassed ? 16 : -10) +
    (finalSentiment === "Positive" ? 8 : finalSentiment === "Neutral" ? 3 : -8);

  return {
    id,
    transcript,
    overallScore: clampScore(baseScore),
    sentiment: {
      initial: initialSentiment,
      final: finalSentiment,
    },
    scorecard: {
      greetingPassed,
      deEscalationPassed,
    },
    actionItems: buildActionItems(greetingPassed, deEscalationPassed, finalSentiment),
  };
};

const createMockLiveMessage = (): LiveMessage => {
  const speaker: Speaker = Math.random() > 0.45 ? "Customer" : "Agent";
  const source = speaker === "Customer" ? MOCK_CUSTOMER_MESSAGES : MOCK_AGENT_MESSAGES;

  return {
    speaker,
    text: source[Math.floor(Math.random() * source.length)],
    timestamp: formatTimestamp(),
  };
};

const ViewButton = ({
  isActive,
  children,
  onClick,
}: {
  isActive: boolean;
  children: React.ReactNode;
  onClick: () => void;
}) => (
  <button
    type="button"
    onClick={onClick}
    className={`rounded-full px-4 py-2 text-sm font-medium transition ${
      isActive ? "bg-primary text-primary-foreground" : "bg-white/5 text-muted-foreground hover:bg-white/10 hover:text-foreground"
    }`}
  >
    {children}
  </button>
);

const MetricCard = ({ label, value, hint }: { label: string; value: string; hint: string }) => (
  <div className="rounded-3xl border border-white/10 bg-black/10 p-5">
    <p className="font-mono text-[10px] uppercase tracking-[0.28em] text-muted-foreground">{label}</p>
    <p className="mt-2 font-display text-3xl font-bold">{value}</p>
    <p className="mt-2 text-xs text-muted-foreground">{hint}</p>
  </div>
);

const QADashboardPage = () => {
  const { toast } = useToast();

  const [currentView, setCurrentView] = useState<DashboardView>("History");
  const [callDatabase, setCallDatabase] = useState<CallReport[]>(HISTORY_SEED);
  const [selectedReportId, setSelectedReportId] = useState<string | null>(HISTORY_SEED[0]?.id ?? null);
  const [analysisInput, setAnalysisInput] = useState("");
  const [processingStatus, setProcessingStatus] = useState<string[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [liveTranscript, setLiveTranscript] = useState<LiveMessage[]>([]);
  const [isLiveRunning, setIsLiveRunning] = useState(false);
  const [liveSuggestion, setLiveSuggestion] = useState("Tip: Start the monitor to receive live coaching.");

  const bottomRef = useRef<HTMLDivElement | null>(null);
  const processingRunRef = useRef(0);

  const selectedReport = useMemo(
    () => callDatabase.find((report) => report.id === selectedReportId) ?? null,
    [callDatabase, selectedReportId]
  );

  const liveAnalytics = useMemo(() => {
    const agentMessages = liveTranscript.filter((message) => message.speaker === "Agent");
    const customerMessages = liveTranscript.filter((message) => message.speaker === "Customer");
    const countWords = (messages: LiveMessage[]) =>
      messages.reduce((total, message) => total + message.text.trim().split(/\s+/).filter(Boolean).length, 0);

    const agentWords = countWords(agentMessages);
    const customerWords = countWords(customerMessages);
    const totalWords = Math.max(agentWords + customerWords, 1);

    return {
      agentWords,
      customerWords,
      agentRatio: Math.round((agentWords / totalWords) * 100),
      customerRatio: Math.round((customerWords / totalWords) * 100),
      totalMessages: liveTranscript.length,
    };
  }, [liveTranscript]);

  const processTranscript = useCallback(async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed) {
      toast({
        title: "Transcript required",
        description: "Paste a transcript before starting QA analysis.",
        variant: "destructive",
      });
      return;
    }

    const runId = Date.now();
    processingRunRef.current = runId;
    setIsProcessing(true);
    setProcessingStatus([]);

    for (const step of WAIT_STEPS) {
      setProcessingStatus((previous) => [...previous, step.label]);
      await wait(step.delay);

      if (processingRunRef.current !== runId) {
        return;
      }
    }

    const report = createCallReport(`CALL-${Date.now()}`, trimmed);
    setCallDatabase((previous) => [report, ...previous]);
    setSelectedReportId(report.id);
    setCurrentView("ReportDetail");
    setIsProcessing(false);
    setProcessingStatus((previous) => [...previous, "Complete. Report saved to history."]);
    toast({
      title: "QA analysis complete",
      description: `Saved ${report.id} to call history.`,
    });
  }, [toast]);

  const saveLiveSession = useCallback(() => {
    if (!liveTranscript.length) {
      toast({
        title: "Nothing to save",
        description: "Run the live monitor first so there is a session to store.",
        variant: "destructive",
      });
      return;
    }

    const transcript = liveTranscript.map((message) => `${message.speaker}: ${message.text}`).join(" ");
    const report = createCallReport(`LIVE-${Date.now()}`, transcript);

    setCallDatabase((previous) => [report, ...previous]);
    setSelectedReportId(report.id);
    setCurrentView("ReportDetail");
    setIsLiveRunning(false);
    setLiveSuggestion("Tip: Live session saved. Review the report and action items.");

    toast({
      title: "Live session saved",
      description: `Saved ${report.id} to call history.`,
    });
  }, [liveTranscript, toast]);

  useEffect(() => {
    if (!isLiveRunning) {
      return undefined;
    }

    const intervalId = window.setInterval(() => {
      setLiveTranscript((previous) => [...previous, createMockLiveMessage()]);
    }, 2000);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [isLiveRunning]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [liveTranscript]);

  useEffect(() => {
    const lastThreeMessages = liveTranscript.slice(-3);

    if (lastThreeMessages.length === 3 && lastThreeMessages.every((message) => message.speaker === "Customer")) {
      setLiveSuggestion("Tip: Express empathy and let the customer finish.");
      return;
    }

    if (!liveTranscript.length) {
      setLiveSuggestion("Tip: Start the monitor to receive live coaching.");
      return;
    }

    const lastSpeaker = liveTranscript[liveTranscript.length - 1]?.speaker;
    setLiveSuggestion(
      lastSpeaker === "Agent"
        ? "Tip: Pause briefly and invite the customer to respond."
        : "Tip: Reflect the concern and confirm the next step."
    );
  }, [liveTranscript]);

  useEffect(() => {
    return () => {
      processingRunRef.current = -1;
    };
  }, []);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto max-w-7xl px-6 py-8 md:px-8">
        <div className="mb-8 flex flex-col gap-4 rounded-[32px] border border-white/10 bg-black/10 p-6 md:flex-row md:items-center md:justify-between">
          <div className="flex items-start gap-4">
            <div className="flex h-14 w-14 items-center justify-center rounded-3xl bg-primary/10">
              <Shield className="h-7 w-7 text-primary" />
            </div>
            <div>
              <p className="font-mono text-[11px] uppercase tracking-[0.3em] text-primary">AI Call Monitor & QA Dashboard</p>
              <h1 className="mt-2 font-display text-4xl font-bold">Review calls, simulate live coaching, and save QA reports.</h1>
              <p className="mt-3 max-w-3xl text-sm leading-relaxed text-muted-foreground">
                This dashboard keeps history, runs async QA analysis, simulates a live monitor, and turns live sessions into saved call reports.
              </p>
            </div>
          </div>
          <Button asChild variant="outline">
            <Link to="/">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Home
            </Link>
          </Button>
        </div>

        <div className="mb-8 flex flex-wrap gap-3">
          <ViewButton isActive={currentView === "History"} onClick={() => setCurrentView("History")}>
            <span className="inline-flex items-center gap-2">
              <History className="h-4 w-4" />
              History
            </span>
          </ViewButton>
          <ViewButton isActive={currentView === "NewAnalysis"} onClick={() => setCurrentView("NewAnalysis")}>
            <span className="inline-flex items-center gap-2">
              <Sparkles className="h-4 w-4" />
              New Analysis
            </span>
          </ViewButton>
          <ViewButton isActive={currentView === "LiveMonitor"} onClick={() => setCurrentView("LiveMonitor")}>
            <span className="inline-flex items-center gap-2">
              <Bot className="h-4 w-4" />
              Live Monitor
            </span>
          </ViewButton>
          {selectedReport && (
            <ViewButton isActive={currentView === "ReportDetail"} onClick={() => setCurrentView("ReportDetail")}>
              Report Detail
            </ViewButton>
          )}
        </div>

        {currentView === "History" && (
          <section className="grid gap-6 lg:grid-cols-[0.72fr_1.28fr]">
            <div className="rounded-[32px] border border-white/10 bg-black/10 p-6">
              <h2 className="font-display text-2xl font-bold">Saved call reports</h2>
              <p className="mt-2 text-sm text-muted-foreground">Click any report to open its detailed QA summary.</p>
              <div className="mt-6 space-y-3">
                {callDatabase.map((report) => (
                  <button
                    key={report.id}
                    type="button"
                    onClick={() => {
                      setSelectedReportId(report.id);
                      setCurrentView("ReportDetail");
                    }}
                    className="w-full rounded-3xl border border-white/10 bg-white/5 p-4 text-left transition hover:border-primary/40 hover:bg-white/10"
                  >
                    <div className="flex items-center justify-between gap-4">
                      <div>
                        <p className="font-mono text-[11px] uppercase tracking-[0.26em] text-primary">{report.id}</p>
                        <p className="mt-2 line-clamp-2 text-sm text-muted-foreground">{report.transcript}</p>
                      </div>
                      <div className="text-right">
                        <p className="text-3xl font-bold">{report.overallScore}</p>
                        <p className="text-xs text-muted-foreground">QA score</p>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            <div className="rounded-[32px] border border-white/10 bg-black/10 p-6">
              <h2 className="font-display text-2xl font-bold">Quick summary</h2>
              <div className="mt-6 grid gap-4 md:grid-cols-3">
                <MetricCard label="Reports" value={String(callDatabase.length).padStart(2, "0")} hint="Saved call reports in history." />
                <MetricCard
                  label="Top Score"
                  value={`${Math.max(...callDatabase.map((report) => report.overallScore), 0)}`}
                  hint="Highest QA result currently stored."
                />
                <MetricCard
                  label="Needs Help"
                  value={`${callDatabase.filter((report) => report.overallScore < 75).length}`}
                  hint="Calls that likely need coaching review."
                />
              </div>
              <div className="mt-6 rounded-3xl border border-white/10 bg-white/5 p-5">
                <p className="text-sm leading-relaxed text-muted-foreground">
                  Use <span className="font-semibold text-foreground">New Analysis</span> to process a transcript asynchronously, or run
                  <span className="font-semibold text-foreground"> Live Monitor</span> to simulate a call and save it directly as a QA report.
                </p>
              </div>
            </div>
          </section>
        )}

        {currentView === "NewAnalysis" && (
          <section className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
            <div className="rounded-[32px] border border-white/10 bg-black/10 p-6">
              <h2 className="font-display text-2xl font-bold">Run asynchronous QA analysis</h2>
              <p className="mt-2 text-sm text-muted-foreground">
                Paste a call transcript and the dashboard will simulate a multi-step QA workflow before saving the report.
              </p>
              <textarea
                value={analysisInput}
                onChange={(event) => setAnalysisInput(event.target.value)}
                placeholder="Paste the call transcript here..."
                className="mt-6 min-h-[280px] w-full rounded-3xl border border-white/10 bg-background/50 px-5 py-4 text-sm outline-none transition focus:border-primary/50"
              />
              <div className="mt-6 flex flex-wrap gap-3">
                <Button onClick={() => void processTranscript(analysisInput)} disabled={isProcessing}>
                  <Sparkles className="mr-2 h-4 w-4" />
                  {isProcessing ? "Processing..." : "Process Transcript"}
                </Button>
                <Button
                  variant="outline"
                  onClick={() => {
                    setAnalysisInput("");
                    setProcessingStatus([]);
                  }}
                >
                  Clear
                </Button>
              </div>
            </div>

            <div className="rounded-[32px] border border-white/10 bg-black/10 p-6">
              <h2 className="font-display text-2xl font-bold">Processing status</h2>
              <p className="mt-2 text-sm text-muted-foreground">Status updates appear in sequence while the mock QA engine runs.</p>
              <div className="mt-6 space-y-3">
                {processingStatus.length ? (
                  processingStatus.map((status, index) => (
                    <div key={`${status}-${index}`} className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm">
                      {status}
                    </div>
                  ))
                ) : (
                  <div className="rounded-2xl border border-dashed border-white/10 px-4 py-6 text-sm text-muted-foreground">
                    No analysis running yet.
                  </div>
                )}
              </div>
            </div>
          </section>
        )}

        {currentView === "LiveMonitor" && (
          <section className="grid gap-6 lg:grid-cols-[0.92fr_1.08fr]">
            <div className="space-y-6">
              <div className="rounded-[32px] border border-white/10 bg-black/10 p-6">
                <h2 className="font-display text-2xl font-bold">Live monitor simulation</h2>
                <p className="mt-2 text-sm text-muted-foreground">
                  Every two seconds, the monitor appends a mock agent or customer message and updates coaching suggestions in real time.
                </p>
                <div className="mt-6 flex flex-wrap gap-3">
                  <Button onClick={() => setIsLiveRunning((previous) => !previous)}>
                    <Play className="mr-2 h-4 w-4" />
                    {isLiveRunning ? "Pause" : "Start"}
                  </Button>
                  <Button variant="outline" onClick={saveLiveSession}>
                    <Save className="mr-2 h-4 w-4" />
                    Save Session
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => {
                      setIsLiveRunning(false);
                      setLiveTranscript([]);
                      setLiveSuggestion("Tip: Start the monitor to receive live coaching.");
                    }}
                  >
                    Reset
                  </Button>
                </div>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <MetricCard
                  label="Talk / Listen"
                  value={`${liveAnalytics.agentRatio}% / ${liveAnalytics.customerRatio}%`}
                  hint="Agent vs. customer share of spoken words."
                />
                <MetricCard
                  label="Messages"
                  value={String(liveAnalytics.totalMessages).padStart(2, "0")}
                  hint="Live messages captured in this session."
                />
              </div>

              <div className="rounded-[32px] border border-white/10 bg-black/10 p-6">
                <p className="font-mono text-[10px] uppercase tracking-[0.28em] text-primary">Live suggestion</p>
                <p className="mt-3 text-base leading-relaxed">{liveSuggestion}</p>
              </div>
            </div>

            <div className="rounded-[32px] border border-white/10 bg-black/10 p-6">
              <div className="flex items-center gap-3">
                <MessageSquareText className="h-5 w-5 text-primary" />
                <h2 className="font-display text-2xl font-bold">Live transcript</h2>
              </div>
              <div className="mt-6 h-[520px] overflow-y-auto rounded-3xl border border-white/10 bg-background/40 p-4">
                <div className="space-y-3">
                  {liveTranscript.length ? (
                    liveTranscript.map((message, index) => (
                      <div
                        key={`${message.timestamp}-${index}`}
                        className={`rounded-3xl px-4 py-3 ${
                          message.speaker === "Agent"
                            ? "bg-primary/10 text-foreground"
                            : "bg-white/5 text-foreground"
                        }`}
                      >
                        <div className="flex items-center justify-between gap-4">
                          <p className="text-sm font-semibold">{message.speaker}</p>
                          <p className="text-xs text-muted-foreground">{message.timestamp}</p>
                        </div>
                        <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{message.text}</p>
                      </div>
                    ))
                  ) : (
                    <div className="rounded-3xl border border-dashed border-white/10 px-4 py-6 text-sm text-muted-foreground">
                      No live messages yet. Start the simulation to populate the transcript.
                    </div>
                  )}
                  <div ref={bottomRef} />
                </div>
              </div>
            </div>
          </section>
        )}

        {currentView === "ReportDetail" && selectedReport && (
          <section className="rounded-[32px] border border-white/10 bg-black/10 p-6">
            <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
              <div>
                <p className="font-mono text-[11px] uppercase tracking-[0.3em] text-primary">{selectedReport.id}</p>
                <h2 className="mt-2 font-display text-3xl font-bold">Detailed QA report</h2>
              </div>
              <Button variant="outline" onClick={() => setCurrentView("History")}>
                Back to History
              </Button>
            </div>

            <div className="mt-6 grid gap-4 md:grid-cols-4">
              <MetricCard label="Overall Score" value={`${selectedReport.overallScore}`} hint="Final QA score for the call." />
              <MetricCard label="Initial Mood" value={selectedReport.sentiment.initial} hint="How the call started emotionally." />
              <MetricCard label="Final Mood" value={selectedReport.sentiment.final} hint="How the call ended emotionally." />
              <MetricCard
                label="Greeting"
                value={selectedReport.scorecard.greetingPassed ? "Passed" : "Missed"}
                hint="Whether the agent opened the call well."
              />
            </div>

            <div className="mt-6 grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
              <div className="rounded-3xl border border-white/10 bg-white/5 p-5">
                <h3 className="font-display text-xl font-bold">Transcript</h3>
                <p className="mt-4 whitespace-pre-wrap text-sm leading-relaxed text-muted-foreground">{selectedReport.transcript}</p>
              </div>

              <div className="space-y-6">
                <div className="rounded-3xl border border-white/10 bg-white/5 p-5">
                  <h3 className="font-display text-xl font-bold">Scorecard</h3>
                  <div className="mt-4 space-y-3 text-sm">
                    <div className="flex items-center justify-between rounded-2xl bg-background/40 px-4 py-3">
                      <span>Greeting</span>
                      <span className={selectedReport.scorecard.greetingPassed ? "text-primary" : "text-destructive"}>
                        {selectedReport.scorecard.greetingPassed ? "Passed" : "Needs work"}
                      </span>
                    </div>
                    <div className="flex items-center justify-between rounded-2xl bg-background/40 px-4 py-3">
                      <span>De-escalation</span>
                      <span className={selectedReport.scorecard.deEscalationPassed ? "text-primary" : "text-destructive"}>
                        {selectedReport.scorecard.deEscalationPassed ? "Passed" : "Needs work"}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="rounded-3xl border border-white/10 bg-white/5 p-5">
                  <h3 className="font-display text-xl font-bold">Action items</h3>
                  <ul className="mt-4 space-y-3 text-sm text-muted-foreground">
                    {selectedReport.actionItems.map((item) => (
                      <li key={item} className="rounded-2xl bg-background/40 px-4 py-3">
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </section>
        )}
      </div>
    </div>
  );
};

export default QADashboardPage;
