#include "core/AnswerKey.hpp"
#include <sstream>
#include <iostream>
#include <algorithm>

// String'i CSV mantığıyla parçalama (Virgülle ayrılmış değerler)
static std::vector<std::string> splitCSV(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        // Boşlukları temizle (trim)
        size_t a = 0, b = tok.size();
        while (a < b && isspace((unsigned char)tok[a])) a++;
        while (b > a && isspace((unsigned char)tok[b - 1])) b--;
        out.push_back(tok.substr(a, b - a));
    }
    return out;
}

void AnswerKey::loadAnswerKey(const std::vector<QuestionAnswer>& keys) {
    keyMap_.clear();
    for (const auto& k : keys) {
        keyMap_[k.subject][k.questionNumber] = k.correctAnswer;
    }
}

AnswerKey::ScoreResult AnswerKey::calculateScore(
    const std::map<std::string, std::string>& studentAnswersCsv) 
{
    ScoreResult res;
    
    // Her bir dersi (Subject) tek tek puanla
    for (const auto& pair : keyMap_) {
        std::string subject = pair.first;
        const auto& correctMap = pair.second; // Soru No -> Doğru Şık

        // İstatistikleri sıfırla
        SubjectStat stat;
        
        // Öğrencinin bu derse ait cevaplarını al (Örn: "A,B,-,D...")
        std::string rawAnswers = "";
        if (studentAnswersCsv.find(subject) != studentAnswersCsv.end()) {
            rawAnswers = studentAnswersCsv.at(subject);
        }

        // CSV'yi parçala (vector string: ["A", "B", "-", "D"])
        std::vector<std::string> tokens = splitCSV(rawAnswers);

        // O dersteki en büyük soru numarasını bul (Döngü limiti için)
        int maxQ = -1;
        for (const auto& q : correctMap) {
            if (q.first > maxQ) maxQ = q.first;
        }
        
        // Soruları Puanla
        for (int qIdx = 0; qIdx <= maxQ; ++qIdx) {
            // 1. Doğru cevabı bul
            char correct = '-';
            if (correctMap.find(qIdx) != correctMap.end()) {
                correct = correctMap.at(qIdx);
            }

            // 2. Öğrenci cevabını bul
            char student = '-';
            if (qIdx < (int)tokens.size()) {
                std::string t = tokens[qIdx];
                if (!t.empty()) student = t[0]; // İlk karakteri al
            }

            // 3. Karşılaştır
            if (student == 'X' || student == '-' || student == ' ' || student == '?') {
                stat.empty++;
            } 
            else if (student == correct) {
                stat.correct++;
            } 
            else {
                stat.wrong++;
            }
        }

        // Net Hesabı (3 yanlış 1 doğruyu götürür)
        stat.net = stat.correct - (stat.wrong / 3.0);

        // Genel Toplama Ekle
        res.totalQuestions += (stat.correct + stat.wrong + stat.empty);
        res.totalCorrect += stat.correct;
        res.totalWrong += stat.wrong;
        res.totalEmpty += stat.empty;
        
        // Toplam puan (Basitçe netlerin toplamı veya 100 üzerinden formülize edilebilir)
        // Burada basitçe Toplam Net'i puan olarak alıyoruz.
        res.totalScore += stat.net;

        // Detaylara kaydet
        res.subjectDetails[subject] = stat;
    }

    return res;
}