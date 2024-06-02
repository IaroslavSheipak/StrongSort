#include <iostream>
#include <string>
#include <numeric>
#include <iterator>
#include <map>
#include <set>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <rectangular_lsap.h>

#include "./nlohmann/json.hpp"
using json = nlohmann::json;

#include "tracker.h"

using namespace std;
using namespace Eigen;
using namespace cv;

namespace strongsort {

using real = float;

class KalmanFilter {
public:
    static constexpr int ndim = 4;
    using FullVector = Matrix<real, ndim * 2, 1>;
    using FullMatrix = Matrix<real, ndim * 2, ndim * 2>;
    using VectorN = Matrix<real, ndim, 1>;
    using MatrixN = Matrix<real, ndim, ndim>;

    KalmanFilter();

    pair<FullVector, FullMatrix> initiate(const VectorN& measurement) noexcept;
    pair<FullVector, FullMatrix> predict(const FullVector& mean, const FullMatrix& covariance) noexcept;
    pair<VectorN, MatrixN> project(const FullVector& mean, const FullMatrix& covariance, real confidence = 0.0) const noexcept;
    pair<FullVector, FullMatrix> update(const FullVector& mean, const FullMatrix& covariance, const VectorN& measurement, real confidence = 0.0);
    VectorX<real> gatingDistance(const FullVector& mean, const FullMatrix& covariance, const Matrix<real, Dynamic, ndim>& measurements, bool onlyPosition = false) const;

private:
    FullMatrix motionMat = FullMatrix::Identity();
    Matrix<real, ndim, ndim * 2> updateMat = Matrix<real, ndim, ndim * 2>::Identity();
    real stdWeightPosition = 1.0 / 20;
    real stdWeightVelocity = 1.0 / 160;
};

KalmanFilter::KalmanFilter() {
    for (int i = 0; i < ndim; ++i) {
        motionMat(i, ndim + i) = 1.0;
    }
}

pair<KalmanFilter::FullVector, KalmanFilter::FullMatrix> KalmanFilter::initiate(const VectorN& measurement) noexcept {
    FullVector mean;
    mean << measurement, VectorN::Zero();
    
    Vector<real, ndim * 2> std;
    std << 2 * stdWeightPosition * measurement[0], 
           2 * stdWeightPosition * measurement[1],
           1 * measurement[2],
           2 * stdWeightPosition * measurement[3],
           10 * stdWeightVelocity * measurement[0],
           10 * stdWeightVelocity * measurement[1],
           0.1 * measurement[2],
           10 * stdWeightVelocity * measurement[3];
    
    FullMatrix covariance = std.array().square().matrix().asDiagonal();
    return make_pair(mean, covariance);
}

pair<KalmanFilter::FullVector, KalmanFilter::FullMatrix> KalmanFilter::predict(const FullVector& mean, const FullMatrix& covariance) noexcept {
    Array<real, ndim * 2, 1> std;
    std << stdWeightPosition * mean[0],
           stdWeightPosition * mean[1],
           1 * mean[2],
           stdWeightPosition * mean[3],
           stdWeightVelocity * mean[0],
           stdWeightVelocity * mean[1],
           0.1 * mean[2],
           stdWeightVelocity * mean[3];

    FullMatrix motionCov = std.square().matrix().asDiagonal();
    FullVector nextMean = motionMat * mean;
    FullMatrix nextCovariance = motionMat * covariance * motionMat.transpose() + motionCov;
    return make_pair(nextMean, nextCovariance);
}

pair<KalmanFilter::VectorN, KalmanFilter::MatrixN> KalmanFilter::project(const FullVector& mean, const FullMatrix& covariance, real confidence) const noexcept {
    Array<real, ndim, 1> std;
    std << stdWeightPosition * mean[3],
           stdWeightPosition * mean[3],
           0.1,
           stdWeightPosition * mean[3];
    std = (1 - confidence) * std;
    MatrixN innovationCov = std.square().matrix().asDiagonal();
    return make_pair(updateMat * mean, updateMat * covariance * updateMat.transpose() + innovationCov);
}

pair<KalmanFilter::FullVector, KalmanFilter::FullMatrix> KalmanFilter::update(const FullVector& mean, const FullMatrix& covariance, const VectorN& measurement, real confidence) {
    auto [projectedMean, projectedCov] = project(mean, covariance, confidence);

    Matrix<real, 2 * ndim, ndim> kalmanGain = projectedCov.ldlt().solve(updateMat * covariance.transpose()).transpose();

    VectorN innovation = measurement - projectedMean;
    FullVector newMean = mean + kalmanGain * innovation;
    FullMatrix newCovariance = covariance - kalmanGain * projectedCov * kalmanGain.transpose();
    return make_pair(newMean, newCovariance);
}

VectorX<real> KalmanFilter::gatingDistance(const FullVector& mean, const FullMatrix& covariance, const Matrix<real, Dynamic, ndim>& measurements, bool onlyPosition) const {
    auto [pMean, pCovariance] = project(mean, covariance);

    if (onlyPosition) {
        auto posMean = pMean.head(2);
        auto posCov = pCovariance.topLeftCorner(2, 2);
        auto posMeasurements = measurements.leftCols(2);
        auto d = posMeasurements.transpose().colwise() - posMean;
        return posCov.llt().matrixL().solve(d).array().square().colwise().sum();
    } else {
        auto d = measurements.transpose().colwise() - pMean;
        return pCovariance.llt().matrixL().solve(d).array().square().colwise().sum();
    }
}

class NearestNeighborDistanceMetric {
public:
    float matchingThreshold;
    enum class Type {
        euclidean,
        cosine
    };

    NearestNeighborDistanceMetric(Type metric, float matchingThreshold, int budget = -1)
        : type(metric), matchingThreshold(matchingThreshold), budget(budget) {}

    void partialFit(const Matrix<real, Dynamic, Eigen::Dynamic>& features, const vector<int>& targets, const vector<int>& activeTargets) {
        assert(features.rows() == static_cast<int>(targets.size()));
        for (size_t i = 0; i < targets.size(); ++i) {
            auto feature = features.row(i);
            auto target = targets[i];
            auto& featureList = samples[target];
            if (budget > 0 && static_cast<int>(featureList.size()) >= budget) {
                std::rotate(featureList.begin(), featureList.begin() + 1, featureList.end());
                featureList.back() = feature;
            } else {
                featureList.push_back(feature);
            }
        }
        map<int, vector<Feature>> newSamples;
        for (const auto& t : activeTargets) {
            newSamples[t] = samples[t];
        }
        samples = std::move(newSamples);
    }

    Matrix<real, Dynamic, Dynamic> distance(const Matrix<real, Dynamic, Eigen::Dynamic>& features, const vector<int>& targets) {
        Matrix<real, Dynamic, Dynamic> costMatrix(targets.size(), features.rows());
        Matrix<real, Dynamic, Eigen::Dynamic> sampleMatrix;

        for (size_t i = 0; i < targets.size(); ++i) {
            const auto& sample = samples[targets[i]];
            sampleMatrix.resize(sample.size(), features.cols());
            for (size_t j = 0; j < sample.size(); ++j) {
                sampleMatrix.row(j) = sample[j];
            }
            costMatrix.row(i) = metric(sampleMatrix, features);
        }
        return costMatrix;
    }

private:
    static Matrix<real, Dynamic, Eigen::Dynamic> normalized(const Matrix<real, Dynamic, Eigen::Dynamic>& x, float eps = 1e-12) noexcept {
        auto norm = x.rowwise().norm().array().max(eps);
        return x.array().colwise() / norm;
    }

    Matrix<real, Dynamic, Dynamic> metric(const Matrix<real, Dynamic, Eigen::Dynamic>& x, const Matrix<real, Dynamic, Eigen::Dynamic>& y) {
        Matrix<real, Dynamic, Dynamic> dist;
        switch (type) {
        case Type::cosine: {
            auto xn = normalized(x);
            auto yn = normalized(y);
            dist = 1.0 - (xn * yn.transpose()).array();
            break;
        }
        case Type::euclidean:
            throw std::runtime_error("Euclidean distance not implemented");
        }
        return dist.colwise().minCoeff();
    }

    Type type;
    int budget;
    map<int, vector<Feature>> samples;
};

struct Detection
{
    Vector4<real> tlwh;
    size_t index;
    float confidence;
    Feature feature;

    Vector4<real> xyah() const noexcept
    {
        Vector4<real> ret = tlwh;
        ret.segment<2>(0) += ret.segment<2>(2) / 2;
        ret[2] /= ret[3];
        return ret;
    }
};

struct Track
{
    enum class State
    {
        Tentative = 1,
        Confirmed = 2,
        Deleted = 3
    };
    uint trackId, classId, lastDetectionIdx;
    int hits = 1, age = 1;
    int timeSinceUpdate = 0;
    float emaAlpha;
    State state = State::Tentative;
    float conf;
    int nInit, maxAge;
    Feature feature;
    KalmanFilter kf;
    KalmanFilter::FullVector mean;
    KalmanFilter::FullMatrix covariance;

    Track(const Eigen::Vector4<real> &detection, int trackId, int classId, float conf, int nInit, int maxAge, float emaAlpha,
          const Feature &feature) :
        trackId(trackId), classId(classId), emaAlpha(emaAlpha), conf(conf), nInit(nInit), maxAge(maxAge), feature(feature / feature.norm())
    {
        tie(mean, covariance) = kf.initiate(detection);
    }

    inline void incrementAge() noexcept 
    {
        ++age;
        ++timeSinceUpdate;
    }

    void markMissed() noexcept
    {
        if (state == State::Tentative || timeSinceUpdate > maxAge)
            state = State::Deleted;
    }

    void predict() noexcept
    {
        tie(mean, covariance) = kf.predict(mean, covariance);
        incrementAge();
    }

    TrackedBox result(int width, int height) const noexcept
    {
        Vector4<real> r = tlwh();
        r.segment<2>(2) += r.segment<2>(0); // Convert to (x, y, x+w, y+h)
        return TrackedBox{
            clip(r[0] / width), clip(r[1] / height),
            clip(r[2] / width), clip(r[3] / height),
            trackId, classId, lastDetectionIdx, conf, timeSinceUpdate
        };
    }
    TrackedBox result(const cv::Size &imageSize) const noexcept
    {
        return result(imageSize.width, imageSize.height);
    }

    Vector4<real> tlwh() const noexcept
    {
        Vector4<real> ret = mean.segment<4>(0);
        ret[2] *= ret[3]; // Width
        ret.segment<2>(0) -= ret.segment<2>(2) / 2; // Top-left x, y
        return ret;
    }

    void update(const Detection &detection, int classId, float conf)
    {
        this->conf = conf;
        this->classId = classId;
        this->lastDetectionIdx = detection.index;
        tie(mean, covariance) = kf.update(mean, covariance, detection.xyah(), detection.confidence);

        Feature detectionFeature = detection.feature / detection.feature.norm();
        Feature smoothFeature = emaAlpha * feature + (1 - emaAlpha) * detectionFeature;
        feature = smoothFeature / smoothFeature.norm();

        ++hits;
        timeSinceUpdate = 0;
        if (state == State::Tentative && hits >= nInit) {
            state = State::Confirmed;
        }
    }
    
};

};