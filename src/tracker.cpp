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


constexpr float INFTY_COST = 1e+5;

pair<vector<int64_t>, vector<int64_t>> linearSumAssignment(const MatrixX<real> &costMatrix)
{
    int maximize = 0;

    auto numRows = costMatrix.rows();
    auto numCols = costMatrix.cols();
    auto dim = min(numRows, numCols);
    vector<int64_t> a(dim), b(dim);
    Matrix<double, Dynamic, Dynamic, RowMajor> cm = costMatrix.cast<double>();

    int ret = solve_rectangular_linear_sum_assignment(
      numRows, numCols, cm.data(), false, a.data(), b.data());
    if (ret == RECTANGULAR_LSAP_INFEASIBLE)
        throw runtime_error("cost matrix is infeasible");
    else if (ret == RECTANGULAR_LSAP_INVALID)
        throw runtime_error("matrix contains invalid numeric entries");
    return make_pair(a, b);
}


template<typename DistanceMetric>
tuple<vector<pair<int, int>>, vector<int>, vector<int>> minCostMatching(
            DistanceMetric distanceMetric,
            float maxDistance,
            const vector<Track> &tracks,
            const vector<Detection> &detections,
            const vector<int> &trackIndices,
            const vector<int> &detectionIndices)
{

    if (detectionIndices.empty() || trackIndices.empty())
        return make_tuple(vector<pair<int, int>>(), trackIndices, detectionIndices); // Nothing to match.

    auto costMatrix = distanceMetric(tracks, detections, trackIndices, detectionIndices);
    for(auto &value : costMatrix.reshaped())
        if(value > maxDistance)
            value = maxDistance + 1e-5;

    auto [rowIndices, colIndices] = linearSumAssignment(costMatrix);

    vector<pair<int, int>> matches;
    vector<int> unmatchedTracks, unmatchedDetections;

    assert(rowIndices.size() == colIndices.size());

    for (size_t col = 0; col < detectionIndices.size(); ++col)
        if (find(colIndices.begin(), colIndices.end(), col) == colIndices.end())
            unmatchedDetections.push_back(detectionIndices[col]);
    for (size_t row = 0; row < trackIndices.size(); ++row)
        if (find(rowIndices.begin(), rowIndices.end(), row) == rowIndices.end())
            unmatchedTracks.push_back(trackIndices[row]);
    for (auto row = rowIndices.begin(), col = colIndices.begin(); row != rowIndices.end() && col != colIndices.end(); ++row, ++col)
    {
        auto trackIdx = trackIndices[*row];
        auto detectionIdx = detectionIndices[*col];
        if (costMatrix(*row, *col) > maxDistance)
        {
            unmatchedTracks.push_back(trackIdx);
            unmatchedDetections.push_back(detectionIdx);
        }
        else
            matches.push_back(make_pair(trackIdx, detectionIdx));
    }
    return make_tuple(matches, unmatchedTracks, unmatchedDetections);
}

VectorX<real> iou(const Array4<real> &tlwh, const Array<real, Dynamic, 4> &candidates)
{
    auto rows = candidates.rows();
    Array2<real> boxTL = tlwh.block<2, 1>(0, 0), boxBR = boxTL + tlwh.block<2, 1>(2, 0);
    Array<real, Dynamic, 4> tlbr(rows, 4);
    for (size_t i = 0; i < rows; ++i)
    {
        Array4<real> candidate = candidates.row(i);
        auto candidateTL = candidate.block<2, 1>(0, 0);
        tlbr.block<1, 2>(i, 0) = candidateTL.max(boxTL);
        tlbr.block<1, 2>(i, 2) = (candidateTL + candidate.block<2, 1>(2, 0)).min(boxBR);
    }
    auto wh = (tlbr.block(0, 2, rows, 2) - tlbr.block(0, 0, rows, 2)).max(0.0);
    auto areaIntersection = wh.col(0) * wh.col(1);
    auto areaBox = tlwh[2] * tlwh[3];
    auto areaCandidates = candidates.col(2) * candidates.col(3);
    return areaIntersection / (areaBox + areaCandidates - areaIntersection);
}

MatrixX<real> iouCost(
      const vector<Track> &tracks,
      const vector<Detection> &detections,
      const vector<int> &trackIndices,
      const vector<int> &detectionIndices)
 {
    MatrixX<real> costMatrix = MatrixX<real>::Zero(trackIndices.size(), detectionIndices.size());
    for (size_t row = 0; row < trackIndices.size(); ++row)
    {
        int trackIdx = trackIndices[row];
        if (tracks[trackIdx].timeSinceUpdate > 1)
        {
            costMatrix.row(row).fill(INFTY_COST);
            continue;
        }

        auto bbox = tracks[trackIdx].tlwh();
        Matrix<real, Dynamic, 4> candidates(detectionIndices.size(), 4);
        for (size_t i = 0; i < detectionIndices.size(); ++i)
            candidates.row(i) = detections[detectionIndices[i]].tlwh;
        costMatrix.row(row) = 1.0 - iou(bbox, candidates).array();
    }
    return costMatrix;
}

template<typename DistanceMetric>
tuple<vector<pair<int, int>>, vector<int>, vector<int>> matchingCascade(
        DistanceMetric distanceMetric,
        float maxDistance,
        const vector<Track> &tracks,
        const vector<Detection> &detections,
        const vector<int> &trackIndices,
        const vector<int> &detectionIndices)
{
    auto unmatchedDetections = detectionIndices;
    auto match = minCostMatching(distanceMetric, maxDistance, tracks, detections, trackIndices, unmatchedDetections);
    auto matches = get<0>(match);
    unmatchedDetections = get<2>(match);
    vector<int> unmatchedTracks;
    set<int> mf;
    transform(matches.begin(), matches.end(), inserter(mf, mf.begin()), [] (auto &p) { return p.first; });
    set_difference(trackIndices.begin(), trackIndices.end(),
                   mf.begin(), mf.end(),
                   inserter(unmatchedTracks, unmatchedTracks.end()));
    return make_tuple(matches, unmatchedTracks, unmatchedDetections);
}

const float chi2inv95[] = {3.8415, 5.9915, 7.8147, 9.4877, 11.070, 12.592, 14.067, 15.507, 16.919};

void gateCostMatrix(
        Matrix<real, Dynamic, Dynamic> &costMatrix,
        const vector<Track> &tracks,
        const vector<Detection> &detections,
        const vector<int> trackIndices,
        const vector<int> detectionIndices,
        float gatedCost = 100000.0f,
        bool onlyPosition = false)
{
    size_t gatingDim = onlyPosition ? 2 : 4;
    auto gatingThreshold = chi2inv95[gatingDim - 1];
    Matrix<real, Dynamic, KalmanFilter::ndim> measurements(detectionIndices.size(), KalmanFilter::ndim);
    transform(detectionIndices.begin(), detectionIndices.end(), measurements.rowwise().begin(), [&detections] (auto i) { return detections[i].xyah(); });

    for (size_t row = 0; row < trackIndices.size(); ++row)
    {
        auto trackIdx = trackIndices[row];
        auto &track = tracks[trackIdx];
        auto gatingDistance = track.kf.gatingDistance(track.mean, track.covariance, measurements, onlyPosition);
        for (size_t col = 0; col < costMatrix.cols(); ++col)
        {
            auto value = (gatingDistance[col] > gatingThreshold) ? gatedCost : costMatrix(row, col);
            costMatrix(row, col) = 0.995 * value + (1 - 0.995) * gatingDistance[col];
        }
    }
}

class Tracker
{
    int nextId = 1;
    NearestNeighborDistanceMetric metric;
    const real maxIouDistance;
    const int maxAge, nInit;
    const real lambda, emaAlpha, mcLambda;
public:
    std::vector<Track> tracks;

    Tracker(NearestNeighborDistanceMetric && metric,
            real maxIouDistance = 0.9,
            int maxAge = 30,
            int nInit = 3,
            real lambda = 0,
            real emaAlpha = 0.9,
            real mcLambda = 0.995) noexcept :
        metric(metric), maxIouDistance(maxIouDistance), maxAge(maxAge), nInit(nInit), lambda(lambda), emaAlpha(emaAlpha), mcLambda(mcLambda)
    {
    }

    void predict() noexcept
    {
        for(auto &track: tracks)
            track.predict();
    }

    void incrementAges() noexcept
    {
        for (auto &track: tracks)
        {
            track.incrementAge();
            track.markMissed();
        }
    }

    json dumpTracks() const noexcept
    {
        json result = json::array();
        for (const auto &track: tracks)
        {
            json item;
            item["age"] = track.age;
            item["conf"] = track.conf;
            item["hits"] = track.hits;
            vector<real> mean(track.mean.begin(), track.mean.end());
            item["mean"] = mean;
            item["state"] = track.state;
            item["time_since_update"] = track.timeSinceUpdate;
            item["track_id"] = track.trackId;
            result.push_back(item);
        }
        return result;
    }

    void update(const vector<Detection> &detections, const VectorX<int> &classes, const VectorX<real> &confidences)
    {
        // Run matching cascade.
        auto [matches, unmatchedTracks, unmatchedDetections] = match(detections);

        // Update track set.
        for (auto [trackIdx, detectionIdx]: matches)
            tracks[trackIdx].update(detections[detectionIdx], classes[detectionIdx], confidences[detectionIdx]);

        for (auto trackIdx: unmatchedTracks)
            tracks[trackIdx].markMissed();
        for (auto detectionIdx: unmatchedDetections)
            initiateTrack(detections[detectionIdx], classes[detectionIdx], confidences[detectionIdx]);
        erase_if(tracks, [] (const auto &track) { return track.state == Track::State::Deleted; });

        // Update distance metric.
        vector<int> activeTargets, targets;
        for (const auto &track: tracks)
            if (track.state == Track::State::Confirmed) {
                activeTargets.push_back(track.trackId);
                targets.push_back(track.trackId);
            }
        Matrix<real, Dynamic, FEATURE_SIZE> features(targets.size(), FEATURE_SIZE);
        auto dst = features.rowwise().begin();
        for (const auto &track: tracks)
            if (track.state == Track::State::Confirmed)
                *(dst++) = track.feature;
        metric.partialFit(features, targets, activeTargets);
    }
private:
    tuple<vector<pair<int, int>>, vector<int>, vector<int>> match(const vector<Detection> &detections)
    {
        // Split track set into confirmed and unconfirmed tracks.
        vector<int> confirmedTracks, unconfirmedTracks;
        for (size_t i = 0; i < tracks.size(); ++i)
        {
            auto &track = tracks[i];
            if (track.state == Track::State::Confirmed)
                confirmedTracks.push_back(i);
            else
                unconfirmedTracks.push_back(i);
        }

        auto gatedMetric = [this] (const vector<Track> &tracks, const vector<Detection> &dets, const vector<int> &trackIndices, const vector<int> &detectionIndices)
        {
            Matrix<real, Dynamic, FEATURE_SIZE> features(detectionIndices.size(), FEATURE_SIZE);
            transform(detectionIndices.begin(), detectionIndices.end(), features.rowwise().begin(), [&dets] (int i) { return dets[i].feature; });
            vector<int> targets(trackIndices.size());
            transform(trackIndices.begin(), trackIndices.end(), targets.begin(), [&tracks] (int i) { return tracks[i].trackId;});
            auto costMatrix = metric.distance(features, targets);
            gateCostMatrix(costMatrix, tracks, dets, trackIndices, detectionIndices);
            return costMatrix;
        };

        // Associate confirmed tracks using appearance features.
        vector<int> detectionIndices(detections.size());
        iota(detectionIndices.begin(), detectionIndices.end(), 0);
        auto [matchesA, unmatchedTracksA, unmatchedDetections] =
                matchingCascade(gatedMetric, metric.matchingThreshold, tracks, detections, confirmedTracks, detectionIndices);

        // Associate remaining tracks together with unconfirmed tracks using IOU.
        vector<int> iouTrackCandidates = unconfirmedTracks;
        copy_if(unmatchedTracksA.begin(), unmatchedTracksA.end(),
                back_inserter(iouTrackCandidates), [this] (int i) { return tracks[i].timeSinceUpdate == 1; });

        erase_if(unmatchedTracksA, [this] (int i) { return tracks[i].timeSinceUpdate == 1; });

        vector<pair<int, int>> matchesB;
        vector<int> unmatchedTracksB;
        tie(matchesB, unmatchedTracksB, unmatchedDetections) = minCostMatching(
                iouCost, maxIouDistance, tracks,
                detections, iouTrackCandidates, unmatchedDetections);

        auto matches = move(matchesA);
        move(matchesB.begin(), matchesB.end(), back_inserter(matches));

        vector<int> unmatchedTracks;
        set_union(unmatchedTracksA.begin(), unmatchedTracksA.end(),
                  unmatchedTracksB.begin(), unmatchedTracksB.end(),
                  back_inserter(unmatchedTracks));
        return make_tuple(matches, unmatchedTracks, unmatchedDetections);
    }

    void initiateTrack(const Detection &detection, int classId, float conf) noexcept
    {
        tracks.emplace_back(detection.xyah(), nextId, classId, conf, nInit, maxAge, emaAlpha, detection.feature);
        nextId++;
    }
};

json StrongSort::dumpTracks() const noexcept
{
    return tracker->dumpTracks();
}

unordered_set<uint> StrongSort::trackIds() const noexcept
{
    unordered_set<uint> result(tracker->tracks.size());
    for (const Track &t: tracker->tracks)
        result.emplace(t.trackId);
    return result;
}

StrongSort::StrongSort(real maxDist, real maxIouDistance, int maxAge, int nInit, int nnBudget) :
    maxDist(maxDist)
{
    auto &&metric = NearestNeighborDistanceMetric(NearestNeighborDistanceMetric::Type::cosine, maxDist, nnBudget);
    tracker = new Tracker(move(metric), maxIouDistance, maxAge, nInit);
}

std::unique_ptr<StrongSort> StrongSort::fromJson(const json &config)
{
    return make_unique<StrongSort>(
            config.contains("max_dist") ? config["max_dist"].get<real>() : 0.2,
            config.contains("max_iou_distance") ? config["max_iou_distance"].get<real>() : 0.7
        );
}

StrongSort::~StrongSort()
{
    delete tracker;
}

vector<TrackedBox> StrongSort::update(const Matrix<real, Dynamic, 4> &ltwhs,
                                      const VectorX<real> &confidences,
                                      const VectorX<int> &classes,
                                      const Matrix<real, Dynamic, FEATURE_SIZE> &features,
                                      const array<int, 2> &imageSize)
{
    auto [w, h] = imageSize;
    // generate detections
    vector<Detection> detections;
    detections.reserve(ltwhs.rows());
    for (size_t i = 0; i < ltwhs.rows(); ++i)
    {
        auto ltwh = ltwhs.block<1, 4>(i, 0);
        detections.emplace_back(ltwh, i, confidences[i], features.row(i));
    }

    // update tracker
    tracker->predict();
    tracker->update(detections, classes, confidences);

    // output bbox identities
    vector<TrackedBox> outputs;
    for (const auto &track: tracker->tracks)
        if (track.state == Track::State::Confirmed && track.timeSinceUpdate <= 1) {
            auto tb = track.result(w, h);
            if (!tb.empty())
                outputs.push_back(tb);
        }
    return outputs;
}

vector<TrackedBox> StrongSort::update(const vector<DetectedBox> &boxes, const Matrix<real, Dynamic, FEATURE_SIZE> &features, const array<int, 2> &imageSize)
{
    size_t size = boxes.size();
    Matrix<real, Dynamic, 4> ltwhs(size, 4);
    VectorX<real> confidences(size);
    VectorX<int> classes(size);
    const auto [w, h] = imageSize;
    for (size_t i = 0; i < size; ++i)
    {
        const DetectedBox &box = boxes[i];
        Vector4<real> ltwh(box.x1, box.y1, box.x2, box.y2);
        ltwh = ltwh.array().min(1.0).max(0.0);
        ltwh.block<2, 1>(2, 0) -= ltwh.block<2, 1>(0, 0);
        ltwh = ltwh.array() * Array4<real>(w, h, w, h);
        ltwhs.row(i) = ltwh;
        confidences[i] = box.confidence;
        classes[i] = static_cast<int>(box.classId);
    }
    return update(ltwhs, confidences, classes, features, imageSize);
}

void StrongSort::incrementAges() noexcept
{
    tracker->incrementAges();
}


};