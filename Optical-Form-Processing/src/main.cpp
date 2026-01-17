#include <opencv2/opencv.hpp>
#include "PerspectiveCorrector.hpp"
#include "ROIDetector.hpp"
#include "AnswerKey.hpp"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <map>
#include <vector>

using namespace cv;
using namespace std;

/* =========================================================
   SCORE OVERLAY
   ========================================================= */
static void drawScoreOverlay(cv::Mat& frame, const AnswerKey::ScoreResult& score) {
    int boxWidth = 550;
    int startX = frame.cols - boxWidth - 20;
    int startY = 40;
    int lineHeight = 45;

    int boxHeight = 220 + (static_cast<int>(score.subjectDetails.size()) * lineHeight);
    cv::Rect bgRect(startX - 10, 10, boxWidth, boxHeight);
    bgRect = bgRect & cv::Rect(0, 0, frame.cols, frame.rows);

    if (bgRect.area() > 0) {
        cv::Mat roi = frame(bgRect);
        cv::Mat color(roi.size(), CV_8UC3, cv::Scalar(0, 0, 0));
        cv::addWeighted(color, 0.8, roi, 0.2, 0, roi);
    }

    cv::putText(frame, "PUANLAMA DETAYI", cv::Point(startX, startY),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 3);

    startY += 50;

    stringstream ss;
    ss << "PUAN: " << fixed << setprecision(2) << score.totalScore;
    cv::putText(frame, ss.str(), cv::Point(startX, startY),
                cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 0), 4);

    startY += 45;

    ss.str("");
    ss << "TOPLAM: " << score.totalCorrect << " D | "
       << score.totalWrong << " Y | " << score.totalEmpty << " B";
    cv::putText(frame, ss.str(), cv::Point(startX, startY),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

    startY += 40;

    cv::line(frame, cv::Point(startX, startY - 10),
             cv::Point(startX + boxWidth - 20, startY - 10),
             cv::Scalar(200, 200, 200), 2);

    for (const auto& pair : score.subjectDetails) {
        std::string name = pair.first;
        if (name == "matematik") name = "Mat";
        else if (name == "turkce") name = "TR";
        else if (name == "fen") name = "Fen";
        else if (name == "sosyal") name = "Sos";
        else if (name == "ingilizce") name = "Ing";
        else if (name == "din") name = "Din";

        if (!name.empty()) name[0] = static_cast<char>(toupper(name[0]));

        const auto& stat = pair.second;

        ss.str("");
        ss << name << ": " << stat.correct << "D "
           << stat.wrong << "Y "
           << stat.empty << "B "
           << " Net: " << fixed << setprecision(1) << stat.net;

        cv::putText(frame, ss.str(), cv::Point(startX, startY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.85, cv::Scalar(255, 255, 255), 2);

        startY += lineHeight;
    }
}

/* =========================================================
   HELPERS
   ========================================================= */
static void addSubjectKey(std::vector<AnswerKey::QuestionAnswer>& out,
                          const std::string& subject,
                          const std::string& key) {
    for (int i = 0; i < static_cast<int>(key.size()); ++i) {
        out.push_back({subject, i, key[i]}); // 0-based
    }
}

static std::vector<std::string> splitCSV(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        // trim
        size_t a = 0, b = tok.size();
        while (a < b && isspace((unsigned char)tok[a])) a++;
        while (b > a && isspace((unsigned char)tok[b - 1])) b--;
        out.push_back(tok.substr(a, b - a));
    }
    return out;
}

static std::string shortName(const std::string& subject) {
    if (subject == "turkce") return "TR";
    if (subject == "sosyal") return "Sos";
    if (subject == "din") return "Din";
    if (subject == "ingilizce") return "Ing";
    if (subject == "matematik") return "Mat";
    if (subject == "fen") return "Fen";
    return subject;
}

static char getStudentAnswerAt(const std::map<std::string, std::string>& studentAnswersCsv,
                               const std::string& subject,
                               int qIndex0) {
    auto it = studentAnswersCsv.find(subject);
    if (it == studentAnswersCsv.end()) return '-';

    auto tokens = splitCSV(it->second);
    if (qIndex0 < 0 || qIndex0 >= (int)tokens.size()) return '-';

    std::string a = tokens[qIndex0];
    if (a.empty()) return '-';

    char c = a[0];
    if (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');

    if (c == '-' || c == ' ') return '-';
    if (c == 'X') return 'X';
    return c; // A,B,C,D,E
}

/* =========================================================
   IDENTITY OVERLAY (TC / OgrenciNo / AdSoyad)
   ========================================================= */
static std::string safeGet(const std::map<std::string, std::string>& m, const std::string& k) {
    auto it = m.find(k);
    if (it == m.end()) return "";
    return it->second;
}

static void drawIdentityOverlay(cv::Mat& frame,
                                const std::map<std::string, std::string>& studentAnswers,
                                cv::Point origin = {40, 80}) {
    int x = origin.x;
    int y = origin.y;

    std::string tc   = safeGet(studentAnswers, "tc_kimlik");
    std::string no   = safeGet(studentAnswers, "ogrenci_no");
    std::string name = safeGet(studentAnswers, "adi_soyadi");

    // Büyük siyah başlık + bilgiler
    cv::putText(frame, "KIMLIK BILGILERI",
                {x, y},
                cv::FONT_HERSHEY_SIMPLEX,
                1.05, cv::Scalar(0, 0, 255), 4);

    y += 42;

    if (!tc.empty()) {
        cv::putText(frame, "TC: " + tc,
                    {x, y},
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.0, cv::Scalar(0, 0, 255), 4);
        y += 38;
    }

    if (!no.empty()) {
        cv::putText(frame, "NO: " + no,
                    {x, y},
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.0, cv::Scalar(0, 0, 255), 4);
        y += 38;
    }

    if (!name.empty()) {
        cv::putText(frame, "AD: " + name,
                    {x, y},
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.0, cv::Scalar(0, 0, 255), 4);
        y += 38;
    }
}

/* =========================================================
   COMPARISON OVERLAY (Student vs AnswerKey)
   ========================================================= */
static void drawComparisonOverlay(cv::Mat& frame,
                                  const std::map<std::string, std::string>& studentAnswersCsv,
                                  const std::map<std::string, std::map<int, char>>& answerKeyMap,
                                  cv::Point origin = {40, 180},
                                  int maxLinesTotal = 70) {
    int x = origin.x;
    int y = origin.y;

    cv::putText(frame, "CEVAP KARSILASTIRMA (Ogrenci / Dogru)",
                {x, y},
                cv::FONT_HERSHEY_SIMPLEX,
                1.0, cv::Scalar(0, 0, 0), 4);

    y += 40;

    std::vector<std::string> order = {
        "turkce","sosyal","din","ingilizce","matematik","fen"
    };

    int lines = 0;

    for (const auto& subj : order) {
        auto itKey = answerKeyMap.find(subj);
        if (itKey == answerKeyMap.end()) continue;

        cv::putText(frame, shortName(subj) + ":",
                    {x, y},
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.95, cv::Scalar(0, 0, 0), 3);
        y += 30;

        int maxQ = -1;
        for (const auto& qp : itKey->second) maxQ = std::max(maxQ, qp.first);
        int totalQ = maxQ + 1;
        if (totalQ <= 0) { y += 10; continue; }

        for (int qi = 0; qi < totalQ; ++qi) {
            if (lines >= maxLinesTotal) return;

            char correct = '-';
            auto itC = itKey->second.find(qi);
            if (itC != itKey->second.end()) correct = itC->second;

            char student = getStudentAnswerAt(studentAnswersCsv, subj, qi);

            std::string status;
            cv::Scalar color;

            if (student == '-') { status = "BOS";   color = cv::Scalar(0, 0, 0); }
            else if (student == 'X') { status = "MULTI"; color = cv::Scalar(0, 0, 255); }
            else if (student == correct) { status = "DOGRU"; color = cv::Scalar(0, 200, 0); }
            else { status = "YANLIS"; color = cv::Scalar(0, 0, 255); }

            std::stringstream ss;
            ss << "Q" << (qi + 1) << ": " << student << " / " << correct << "  " << status;

            cv::putText(frame, ss.str(),
                        {x + 20, y},
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.80, color, 2);

            y += 24;
            lines++;
        }

        y += 12;
    }
}

/* =========================================================
   MAIN
   ========================================================= */
int main(int argc, char** argv) {
    int camIndex = 0;
    if (argc > 1) camIndex = std::atoi(argv[1]);

    cv::VideoCapture cap(camIndex, cv::CAP_ANY);
    if (!cap.isOpened()) {
        std::cerr << "Kamera acilamadi! Index: " << camIndex << "\n";
        std::cerr << "Deneme: ./omr 0   veya   ./omr 1\n";
        return 1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    cap.set(cv::CAP_PROP_FPS, 30);
    cap.set(cv::CAP_PROP_AUTOFOCUS, 1);

    core::PerspectiveCorrector pc(1600, 2200);

    ROIDetector detector;
    detector.setFillThreshold(0.40);

    // --- ANSWER KEY ---
    AnswerKey answerKey;
    std::vector<AnswerKey::QuestionAnswer> answers;

    addSubjectKey(answers, "turkce",    "CBAABDBCCCCDABABCAAD");
    addSubjectKey(answers, "sosyal",    "BBDABBACADCBAACDDCCD");
    addSubjectKey(answers, "din",       "BABDBABDBACDBBAACBDB");
    addSubjectKey(answers, "ingilizce", "BABDBABDBACDBBAACBDB");
    addSubjectKey(answers, "matematik", "BABDBABDBACDBBAACBDB");
    addSubjectKey(answers, "fen",       "BABDBABDBACDBBAACBDB");

    answerKey.loadAnswerKey(answers);

    // AnswerKeyMap (student vs correct karşılaştırma için)
    std::map<std::string, std::map<int, char>> answerKeyMap;
    for (const auto& qa : answers) {
        answerKeyMap[qa.subject][qa.questionNumber] = qa.correctAnswer;
    }

    bool showDebug = true;
    bool showBubbleDebug = true;
    bool showCompareOverlay = true;
    int rotationMode = 0;

    bool isPaused = false;
    bool recomputeScore = false;

    cv::Mat currentFrame;

    AnswerKey::ScoreResult lastScore;
    std::map<std::string, std::string> lastStudentAnswers;
    cv::Mat omrDebugImage;
    cv::Mat bubbleDebugImage;

    cout << "=== OPTIK FORM OKUYUCU ===\n";
    cout << "P: durdur/sonuc\n";
    cout << "D: perspective debug ac/kapat\n";
    cout << "B: bubble debug ac/kapat\n";
    cout << "C: compare overlay ac/kapat\n";
    cout << "R: rotate\n";
    cout << "+/-: threshold\n";
    cout << "ESC: cikis\n\n";


    // --- PENCERE AYARLARI
    cv::namedWindow("Kamera", cv::WINDOW_NORMAL);
    cv::namedWindow("Form Analizi", cv::WINDOW_NORMAL);
    cv::namedWindow("Bubble Debug", cv::WINDOW_NORMAL);

    cv::resizeWindow("Kamera", 960, 540);       // 1080p'nin yarısı
    cv::resizeWindow("Form Analizi", 480, 640); // Dikey form için uygun oran
    cv::resizeWindow("Bubble Debug", 480, 640); // Dikey form için uygun oran

    while (true) {
        cv::Mat frame;

        // 1) Capture
        if (!isPaused) {
            if (!cap.read(frame) || frame.empty()) break;
            currentFrame = frame.clone();
        } else {
            if (currentFrame.empty()) continue;
            frame = currentFrame.clone();
        }

        // 2) Rotation
        cv::Mat processedFrame = frame.clone();
        if (rotationMode == 1) cv::rotate(frame, processedFrame, cv::ROTATE_90_CLOCKWISE);
        else if (rotationMode == 2) cv::rotate(frame, processedFrame, cv::ROTATE_90_COUNTERCLOCKWISE);
        else if (rotationMode == 3) cv::rotate(frame, processedFrame, cv::ROTATE_180);

        // 3) Perspective + ROI
        auto R = pc.findAndWarp(processedFrame, showDebug);

        cv::Mat displayFrame;
        if (showDebug && !R.debug.empty()) displayFrame = R.debug.clone();
        else displayFrame = processedFrame.clone();

        if (R.ok && !R.warped.empty()) {
            detector.setDebugMode(showBubbleDebug);

            // Canlı okuma (tc_kimlik / ogrenci_no / adi_soyadi dahil)
            lastStudentAnswers = detector.process(R.warped, omrDebugImage);

            bubbleDebugImage = detector.getLastDebugVisualization();
            if (showBubbleDebug && !bubbleDebugImage.empty()) cv::imshow("Bubble Debug", bubbleDebugImage);
            if (!omrDebugImage.empty()) cv::imshow("Form Analizi", omrDebugImage);

            // Pause anında 1 kez skor
            if (isPaused && recomputeScore) {
                lastScore = answerKey.calculateScore(lastStudentAnswers);
                recomputeScore = false;
            }

            if (isPaused) {
                // 1) Score overlay
                if (lastScore.totalQuestions > 0) drawScoreOverlay(displayFrame, lastScore);

                // 2) Kimlik bilgileri (büyük siyah)
                drawIdentityOverlay(displayFrame, lastStudentAnswers, {40, 70});

                // 3) Student vs Correct overlay (aşağıdan başlasın)
                if (showCompareOverlay) {
                    drawComparisonOverlay(displayFrame, lastStudentAnswers, answerKeyMap, {40, 220}, 70);
                }

                cv::rectangle(displayFrame, cv::Point(0, 0),
                              cv::Point(displayFrame.cols, displayFrame.rows),
                              cv::Scalar(0, 0, 255), 6);
                cv::putText(displayFrame, "SONUC EKRANI (Canli icin P)", cv::Point(40, 40),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
            } else {
                cv::putText(displayFrame, "Hizala ve 'P' tusuna bas", cv::Point(40, 40),
                            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            }

        } else {
            if (isPaused) {
                cv::putText(displayFrame, "KAGIT BULUNAMADI!", cv::Point(50, 200),
                            cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
            }
        }

        // 4) Footer info
        string infoText;
        if (rotationMode == 0) infoText = "Rot: OFF";
        else if (rotationMode == 1) infoText = "Rot: 90 CW";
        else if (rotationMode == 2) infoText = "Rot: 90 CCW";
        else if (rotationMode == 3) infoText = "Rot: 180";

        std::stringstream ts;
        ts << fixed << setprecision(2) << detector.getFillThreshold();
        infoText += " | Hassasiyet: " + ts.str();

        cv::putText(displayFrame, infoText, cv::Point(40, displayFrame.rows - 50),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);

        cv::imshow("Kamera", displayFrame);

        // 5) Keys
        int k = cv::waitKey(1) & 0xFF;
        if (k == 27) break;

        if (k == 'p' || k == 'P') {
            isPaused = !isPaused;
            if (isPaused) recomputeScore = true;
        }

        if (k == 'd' || k == 'D') showDebug = !showDebug;

        if (k == 'r' || k == 'R') rotationMode = (rotationMode + 1) % 4;

        if (k == '+' || k == '=') detector.setFillThreshold(detector.getFillThreshold() + 0.05);
        if (k == '-' || k == '_') detector.setFillThreshold(max(0.05, detector.getFillThreshold() - 0.05));

        if (k == 'b' || k == 'B') {
            showBubbleDebug = !showBubbleDebug;
            if (!showBubbleDebug) { try { cv::destroyWindow("Bubble Debug"); } catch (...) {} }
        }

        if (k == 'c' || k == 'C') {
            showCompareOverlay = !showCompareOverlay;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
