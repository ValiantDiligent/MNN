//
//  BatchTest.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/10/10.
//

import Foundation

/// Errors that can occur during batch file testing
enum BatchFileTestError: LocalizedError {
    case unsupportedFileFormat
    case invalidFileContent
    case fileNotFound
    case permissionDenied

    var errorDescription: String? {
        switch self {
        case .unsupportedFileFormat:
            return "Unsupported file format. Please use .txt, .json, or .jsonl files."
        case .invalidFileContent:
            return "Invalid file content. Please check the file format and try again."
        case .fileNotFound:
            return "The selected file could not be found."
        case .permissionDenied:
            return "Permission denied. Unable to access the selected file."
        }
    }
}

/// Represents the type of batch test being performed
enum BatchTestType: String, CaseIterable {
    case text = "text"
    case image = "image"
    case audio = "audio"
    case tier9 = "tier9"
    
    var displayName: String {
        switch self {
        case .text:
            return "Text Test"
        case .image:
            return "Image Test"
        case .audio:
            return "Audio Test"
        case .tier9:
            return "Tier9 AL"
        }
    }
    
    var fileExtension: String {
        switch self {
        case .text:
            return "jsonl"
        case .image:
            return "jsonl"
        case .audio:
            return "jsonl"
        case .tier9:
            return "jsonl"
        }
    }
    
    var localBatchTestFolder: String {
        switch self {
        case .text:
            return "Texts"
        case .image:
            return "Images"
        case .audio:
            return "Audios"
        case .tier9:
            return "Tier9"
        }
    }
}

/// Represents a single test item in a batch
struct BatchTestItem {
    let prompt: String
    let testType: BatchTestType

    /// Creates a BatchTestItem with explicit test type
    /// - Parameters:
    ///   - prompt: The test prompt text
    ///   - testType: The type of batch test
    init(prompt: String, testType: BatchTestType) {
        self.prompt = prompt
        self.testType = testType
    }
}

/// Represents the result of a single test execution
struct BatchTestResult {
    let item: BatchTestItem
    let response: String
    let timestamp: Date
    /// Per-prompt timing collected by the engine; nil when unavailable
    let metrics: [String: NSNumber]?

    init(item: BatchTestItem, response: String, timestamp: Date, metrics: [String: NSNumber]? = nil) {
        self.item = item
        self.response = response
        self.timestamp = timestamp
        self.metrics = metrics
    }
}

/// Available file formats for batch test output
enum BatchTestFileFormat: String, CaseIterable {
    case txt
    case json
    case jsonl

    var displayName: String {
        switch self {
        case .txt:
            return "Text (.txt)"
        case .json:
            return "JSON (.json)"
        case .jsonl:
            return "JSON Lines (.jsonl)"
        }
    }
}

/// Configuration for batch test operations
struct BatchTestConfiguration {
    let outputFormat: BatchTestFileFormat
    let includeTimestamp: Bool
    let includePrompt: Bool

    init(outputFormat: BatchTestFileFormat, includeTimestamp: Bool, includePrompt: Bool) {
        self.outputFormat = outputFormat
        self.includeTimestamp = includeTimestamp
        self.includePrompt = includePrompt
    }
}
