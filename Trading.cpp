#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdlib>

#include <curl/curl.h>
#include <onnxruntime_cxx_api.h>
#include "json.hpp"

using json = nlohmann::json;

// ================= CONFIG =================

const std::string UPSTOX_TOKEN = std::getenv("UPSTOX_ACCESS_TOKEN");
const std::string BASE_URL = "https://api.upstox.com/v2";
const std::string INSTRUMENT_KEY = "NSE_INDEX|Nifty 50";

const float BUY_THRESHOLD = 0.65f;
const float SELL_LOW = 0.45f;
const float SELL_HIGH = 0.55f;

const int MAX_TRADES = 3;
const int COOLDOWN_MIN = 30;

// ================= GLOBAL STATE =================

std::string STRATEGY;
int trades_taken = 0;
std::chrono::system_clock::time_point last_trade;

// ================= CURL =================

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// ================= FETCH DATA =================

json fetch_live_data() {
    CURL* curl = curl_easy_init();
    std::string response;

    std::string url = BASE_URL +
        "/historical-candle/intraday?instrument_key=" +
        INSTRUMENT_KEY + "&interval=5minute&count=100";

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers,
        ("Authorization: Bearer " + UPSTOX_TOKEN).c_str());

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        throw std::runtime_error("Upstox API failed");
    }

    return json::parse(response);
}

// ================= FEATURE ENGINE =================

std::vector<float> create_features(const json& candles) {
    auto last = candles.back();

    float close = last[4];
    float open = last[1];
    float high = last[2];
    float low  = last[3];
    float volume = last[5];

    float log_return = std::log(close / open);
    float candle_body = (close - open) / open;
    float candle_range = (high - low) / open;

    // Simplified feature vector (must match ONNX export order)
    return {
        log_return,
        candle_body,
        candle_range,
        volume
    };
}

// ================= MODEL INFERENCE =================

float predict_probability(
    Ort::Session& session,
    Ort::AllocatorWithDefaultOptions& allocator,
    std::vector<float>& features
) {
    std::vector<int64_t> shape = {1, (int64_t)features.size()};

    Ort::Value input = Ort::Value::CreateTensor<float>(
        allocator,
        features.data(),
        features.size(),
        shape.data(),
        shape.size()
    );

    auto output = session.Run(
        Ort::RunOptions{nullptr},
        &session.GetInputName(0, allocator),
        &input,
        1,
        &session.GetOutputName(0, allocator),
        1
    );

    float* probs = output[0].GetTensorMutableData<float>();
    return probs[1]; // probability of class 1
}

// ================= SIGNAL LOGIC =================

std::string generate_signal(float prob) {
    if (STRATEGY == "BUYING") {
        if (prob > BUY_THRESHOLD)
            return "BUY CALL";
        if (prob < (1.0f - BUY_THRESHOLD))
            return "BUY PUT";
    }

    if (STRATEGY == "SELLING") {
        if (prob >= SELL_LOW && prob <= SELL_HIGH)
            return "SELL ATM";
    }

    return "NO TRADE";
}

// ================= MAIN =================

int main() {
    std::cout << "Select strategy (BUYING / SELLING): ";
    std::cin >> STRATEGY;

    if (STRATEGY != "BUYING" && STRATEGY != "SELLING") {
        std::cerr << "Invalid strategy\n";
        return 1;
    }

    std::cout << "Strategy locked: " << STRATEGY << "\n";

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "trader");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);

    Ort::Session session(env, "model.onnx", opts);
    Ort::AllocatorWithDefaultOptions allocator;

    while (true) {
        try {
            auto data = fetch_live_data();
            auto candles = data["data"]["candles"];

            auto features = create_features(candles);
            float prob = predict_probability(session, allocator, features);
            std::string signal = generate_signal(prob);

            std::cout
                << "SIGNAL | "
                << STRATEGY << " | "
                << signal << " | "
                << "Prob=" << prob
                << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::minutes(5));
    }

    return 0;
}
