#pragma once
#include <string>
#include <vector>
#include <map>

class AnswerKey {
public:
    struct QuestionAnswer {
        std::string subject;
        int questionNumber; // 0-based
        char correctAnswer;
    };

    struct SubjectStat {
        int correct = 0;
        int wrong = 0;
        int empty = 0;
        double net = 0.0;
    };

    struct ScoreResult {
        int totalQuestions = 0;
        int totalCorrect = 0;
        int totalWrong = 0;
        int totalEmpty = 0;
        double totalScore = 0.0;
        std::map<std::string, SubjectStat> subjectDetails;
    };

    void loadAnswerKey(const std::vector<QuestionAnswer>& keys);
    ScoreResult calculateScore(const std::map<std::string, std::string>& studentAnswersCsv);

private:
    std::map<std::string, std::map<int, char>> keyMap_;
};