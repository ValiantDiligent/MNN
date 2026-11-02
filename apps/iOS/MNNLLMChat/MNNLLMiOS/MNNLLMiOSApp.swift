//
//  MNNLLMiOSApp.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2024/12/26.
//

import Darwin
import SwiftUI

@main
struct MNNLLMiOSApp: App {
    init() {
        // DFlash currently requires command replay to be disabled. MetalEnv
        // snapshots environment variables on first use, so this must happen
        // before any model creates a Metal runtime.
        setenv("MNN_METAL_DISABLE_REPLAY", "1", 1)

        UIView.appearance().overrideUserInterfaceStyle = .light

        let savedLanguage = LanguageManager.shared.currentLanguage
        UserDefaults.standard.set([savedLanguage], forKey: "AppleLanguages")
        UserDefaults.standard.synchronize()

        // Headless runs never touch the screen, so the display would sleep and
        // iOS would suspend the process mid-load.
        UIApplication.shared.isIdleTimerDisabled = true
    }

    var body: some Scene {
        WindowGroup {
            if let benchArgs = Tier9BenchArgs.commandLine {
                Tier9BenchView(args: benchArgs)
            } else if CommandLine.arguments.contains("--preset-asr-silent-probe") {
                PresetASRSilentProbeView()
            } else {
                MainTabView()
            }
        }
    }
}

// MARK: - Headless tier9 bench (launch args: --tier9-bench <modelFolder> --tier9-mode batch|stream --tier9-max-new N)

struct Tier9BenchArgs {
    let modelFolder: String
    let mode: String
    let maxNew: Int32
    let configOverride: String?
    let cooldownSeconds: Int

    static var commandLine: Tier9BenchArgs? {
        let args = CommandLine.arguments
        guard let i = args.firstIndex(of: "--tier9-bench"), i + 1 < args.count else { return nil }
        var mode = "batch"
        if let j = args.firstIndex(of: "--tier9-mode"), j + 1 < args.count { mode = args[j + 1] }
        var maxNew: Int32 = 256
        if let k = args.firstIndex(of: "--tier9-max-new"), k + 1 < args.count,
           let v = Int32(args[k + 1]) { maxNew = v }
        var configOverride: String? = nil
        if let m = args.firstIndex(of: "--tier9-config"), m + 1 < args.count { configOverride = args[m + 1] }
        var cooldown = 0
        if let n = args.firstIndex(of: "--tier9-cooldown"), n + 1 < args.count,
           let v = Int(args[n + 1]) { cooldown = v }
        return Tier9BenchArgs(modelFolder: args[i + 1], mode: mode, maxNew: maxNew,
                              configOverride: configOverride, cooldownSeconds: cooldown)
    }
}

private struct Tier9BenchView: View {
    let args: Tier9BenchArgs
    @State private var status = "tier9 bench starting"

    var body: some View {
        Text(status).task {
            UIApplication.shared.isIdleTimerDisabled = true
            status = await Tier9BenchRunner.run(args)
        }
    }
}

enum Tier9BenchRunner {
    static func run(_ args: Tier9BenchArgs) async -> String {
        NSLog("[TIER9] harness start folder=%@ mode=%@ maxNew=%d cooldown=%d", args.modelFolder, args.mode, args.maxNew, args.cooldownSeconds)
        if args.cooldownSeconds > 0 {
            // Sleeping inside the app keeps it foreground so the display stays on
            // and the process is never suspended; idling does not heat the SoC, so
            // every measured round starts from the same thermal state.
            try? await Task.sleep(nanoseconds: UInt64(args.cooldownSeconds) * 1_000_000_000)
            NSLog("[TIER9] cooldown done after %ds", args.cooldownSeconds)
        }
        let fm = FileManager.default
        guard let resourceURL = Bundle.main.resourceURL else { return "no bundle resource url" }
        let modelURL = resourceURL.appendingPathComponent("LocalModel/\(args.modelFolder)")
        guard fm.fileExists(atPath: modelURL.appendingPathComponent("config.json").path) else {
            NSLog("[TIER9] ERROR model missing: %@", modelURL.path)
            return "model missing"
        }

        if let override = args.configOverride {
            let cfgDir = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("MNNConfigs/\(args.modelFolder)")
            try? fm.createDirectory(at: cfgDir, withIntermediateDirectories: true)
            let cfgURL = cfgDir.appendingPathComponent("custom_config.json")
            try? Data(override.utf8).write(to: cfgURL, options: .atomic)
            NSLog("[TIER9] config override written: %@ %@", cfgURL.path, override)
        }

        var promptDir = resourceURL.appendingPathComponent("LocalBatchTest/Tier9")
        if !fm.fileExists(atPath: promptDir.path) {
            promptDir = resourceURL.appendingPathComponent("LocalModel/tier9")
        }
        var prompts: [(String, String)] = []
        let jsonlURL = promptDir.appendingPathComponent("prompt.jsonl")
        if fm.fileExists(atPath: jsonlURL.path),
           let data = try? Data(contentsOf: jsonlURL) {
            for line in String(decoding: data, as: UTF8.self).split(separator: "\n") {
                guard let row = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
                      let text = row["prompt"] as? String else { continue }
                prompts.append(((row["id"] as? String) ?? "prompt\(prompts.count)", text))
            }
        }
        if prompts.isEmpty {
            if !fm.fileExists(atPath: promptDir.path) {
                promptDir = resourceURL.appendingPathComponent("LocalModel/tier9")
            }
            let files = ((try? fm.contentsOfDirectory(at: promptDir, includingPropertiesForKeys: nil)) ?? [])
                .filter { $0.pathExtension == "txt" }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
            for f in files {
                if let text = try? String(contentsOf: f, encoding: .utf8) {
                    prompts.append((f.lastPathComponent, text.trimmingCharacters(in: .whitespacesAndNewlines)))
                }
            }
        }
        guard !prompts.isEmpty else {
            NSLog("[TIER9] ERROR no prompts in %@", promptDir.path)
            return "no prompts"
        }

        let loadStart = Date()
        NSLog("[TIER9] loading model from %@", modelURL.path)
        var holder: LLMInferenceEngineWrapper?
        let ok: Bool = await withCheckedContinuation { (cont: CheckedContinuation<Bool, Never>) in
            holder = LLMInferenceEngineWrapper(modelPath: modelURL.path) { cont.resume(returning: $0) }
        }
        NSLog("[TIER9] load callback returned ok=%d after %.1fs", ok ? 1 : 0, Date().timeIntervalSince(loadStart))
        guard ok, let llm = holder else {
            NSLog("[TIER9] ERROR model load failed")
            return "load failed"
        }
        let loadSeconds = Date().timeIntervalSince(loadStart)
        let cfg = ModelConfigManager(modelPath: modelURL.path)
        if let json = cfg.readConfigAsJSONString() { llm.setConfigWithJSONString(json) }
        llm.setMaxNewTokens(args.maxNew)
        NSLog("[TIER9] start model=%@ mode=%@ maxNew=%d prompts=%d load_s=%.1f backend=%@ precision=%@ threads=%d",
              args.modelFolder, args.mode, args.maxNew, prompts.count, loadSeconds,
              cfg.readBackendType(), cfg.readPrecision(), cfg.readThreadNum())

        var rows: [[String: Any]] = []
        if args.mode == "stream" {
            for (idx, item) in prompts.enumerated() {
                llm.clearChatHistory()
                var out = ""
                llm.processInput(item.1, withOutput: { out += $0 }, showPerformance: true)
                var waitedMs = 0
                while llm.isProcessing() && waitedMs < 600_000 {
                    try? await Task.sleep(nanoseconds: 200_000_000)
                    waitedMs += 200
                }
                var row: [String: Any] = ["i": idx, "file": item.0, "chars": out.count]
                for (key, pattern) in [("prefill_s", "Prefill: ([0-9.]+)s"),
                                       ("prompt_tokens", "Prefill: [0-9.]+s, ([0-9]+) tokens"),
                                       ("prefill_tok_s", "([0-9.]+) tokens/s\\nDecode:"),
                                       ("decode_s", "Decode: ([0-9.]+)s"),
                                       ("decode_tokens", "Decode: [0-9.]+s, ([0-9]+) tokens"),
                                       ("decode_tok_s", "Decode: [0-9.]+s, [0-9]+ tokens, ([0-9.]+) tokens/s"),
                                       ("calls", "calls=([0-9]+)"),
                                       ("tokens_per_call", "tokens/call=([0-9.]+)")] {
                    if let v = firstMatch(pattern, in: out) { row[key] = Double(v) ?? v }
                }
                rows.append(row)
                NSLog("[TIER9_STREAM] i=%d file=%@ %@", idx, item.0, rowString(row))
            }
        } else {
            let batchPrompts = prompts.map { $0.1 }
            let (responses, metrics): ([String], [[String: NSNumber]]) = await withCheckedContinuation { cont in
                llm.processBatchPrompts(batchPrompts, showPerformance: true) { r, m in
                    cont.resume(returning: (r, m))
                }
            }
            for (idx, item) in prompts.enumerated() {
                var row: [String: Any] = ["i": idx, "file": item.0]
                if idx < responses.count {
                    row["chars"] = responses[idx].count
                    row["text"] = String(responses[idx].prefix(120))
                }
                var dict: [String: Any] = [:]
                if idx < metrics.count {
                    for (k, v) in metrics[idx] { dict[k] = v.doubleValue }
                }
                row["metrics"] = dict
                rows.append(row)
                NSLog("[TIER9_BATCH] i=%d file=%@ chars=%d %@", idx, item.0, (row["chars"] as? Int) ?? 0, dictString(dict))
            }
        }

        let summary: [String: Any] = [
            "model": args.modelFolder,
            "mode": args.mode,
            "max_new": Int(args.maxNew),
            "backend": cfg.readBackendType(),
            "precision": cfg.readPrecision(),
            "threads": cfg.readThreadNum(),
            "load_seconds": loadSeconds,
            "rows": rows,
        ]
        let outURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("tier9_bench_\(args.mode).json")
        if let data = try? JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: outURL, options: .atomic)
        }
        NSLog("[TIER9_DONE] mode=%@ rows=%d out=%@", args.mode, rows.count, outURL.path)
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { exit(0) }
        return "tier9 \(args.mode) done: \(rows.count) prompts"
    }

    private static func firstMatch(_ pattern: String, in text: String) -> String? {
        guard let re = try? NSRegularExpression(pattern: pattern) else { return nil }
        let range = NSRange(text.startIndex..., in: text)
        guard let m = re.firstMatch(in: text, range: range), m.numberOfRanges > 1,
              let r = Range(m.range(at: 1), in: text) else { return nil }
        return String(text[r])
    }

    private static func rowString(_ row: [String: Any]) -> String {
        ["prompt_tokens", "prefill_tok_s", "decode_tokens", "decode_tok_s", "calls", "tokens_per_call"]
            .compactMap { key in row[key].map { "\(key)=\($0)" } }
            .joined(separator: " ")
    }

    private static func dictString(_ dict: [String: Any]) -> String {
        ["prompt_tokens", "gen_tokens", "prefill_tok_s", "decode_tok_s", "steps", "tokens_per_step", "accept_len"]
            .compactMap { key in dict[key].map { "\(key)=\($0)" } }
            .joined(separator: " ")
    }
}

private struct PresetASRSilentProbeView: View {
    @State private var status = "X-ASR silent probe"

    var body: some View {
        Text(status).task {
            status = await Task.detached(priority: .userInitiated) {
                await Self.runProbe()
            }.value
        }
    }

    private static func runProbe() async -> String {
        let reportURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("qwen-xasr-device-probe.json")
        let audioURL = Bundle.main.resourceURL?
            .appendingPathComponent("LocalModel/preset_audio/audio (1).wav")
        var report: [String: Any] = [
            "runtime": "sherpa-mnn",
            "model": "x-asr-zipformer2",
            "precision": "mnn-weight-int8-block64",
            "backend": "metal",
            "threads": 2,
        ]
        let service = PresetZipformerASRService()
        do {
            guard let audioURL else {
                throw PresetASRError.missingAsset("LocalModel/preset_audio/audio (1).wav")
            }
            let recognition = try await service.recognize(
                audioURL: audioURL,
                expectedAudioSHA256: "b112c5b9b4503e59b14102be18bba6cc0cb9cd44a86af87c2b75a31e1f06f066"
            )
            report["transcript"] = recognition.transcript
            report["audio_seconds"] = recognition.audioDurationSeconds
            report["initialization_seconds"] = recognition.modelInitializationSeconds
            report["inference_seconds"] = recognition.inferenceSeconds
            report["rtf"] = recognition.realTimeFactor
            report["realtime_multiple"] = recognition.realTimeMultiple
        } catch {
            report["error"] = error.localizedDescription
        }
        service.shutdown()
        if let data = try? JSONSerialization.data(
            withJSONObject: report, options: [.prettyPrinted, .sortedKeys]
        ) {
            try? data.write(to: reportURL, options: .atomic)
        }
        return report["error"] == nil ? "X-ASR silent probe complete" : "X-ASR silent probe failed"
    }
}
