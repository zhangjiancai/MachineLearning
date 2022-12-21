#ifndef CAFFE_SOLVER_HPP_
#define CAFFE_SOLVER_HPP_
#include <boost/function.hpp>

#include <atomic>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/solver_factory.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

/**
  * @brief Enumeration of actions that a client of the Solver may request by
  * implementing the Solver's action request function, which a
  * a client may optionally provide in order to request early termination
  * or saving a snapshot without exiting. In the executable caffe, this
  * mechanism is used to allow the snapshot to be saved when stopping
  * execution with a SIGINT (Ctrl-C).
  */
  namespace SolverAction {
    enum Enum {
      NONE = 0,  // Take no special action.
      STOP = 1,  // Stop training. snapshot_after_train controls whether a
                 // snapshot is created.
      SNAPSHOT = 2  // Take a snapshot, and keep training.
    };
  }

/**
 * @brief Type of a function that returns a Solver Action enumeration.
 */
typedef boost::function<SolverAction::Enum()> ActionCallback;

/**
 * @brief An interface for classes that perform optimization on Net%s.
 *
 * Requires implementation of ApplyUpdate to compute a parameter update
 * given the current state of the Net parameters.
 */
class Solver {
 public:
  explicit Solver(const SolverParameter& param, size_t rank, const Solver* root_solver = NULL);
  explicit Solver(const string& param_file, size_t rank, const Solver* root_solver = NULL);
  void Init();
  void InitTrainNet();
  void InitTestNets();

  // Client of the Solver optionally may call this in order to set the function
  // that the solver uses to see what action it should take (e.g. snapshot or
  // exit training early).
  void SetActionFunction(ActionCallback func);
  SolverAction::Enum GetRequestedAction();
  // The main entry of the solver function. In default, iter will be zero. Pass
  // in a non-zero iter number to resume training for a pre-trained net.
  virtual bool Solve(const char* resume_file = NULL);
  void Solve(const string resume_file) { Solve(resume_file.c_str()); }
  void Step(int iters);
  // The Restore method simply dispatches to one of the
  // RestoreSolverStateFrom___ protected methods. You should implement these
  // methods to restore the state from the appropriate snapshot type.
  void Restore(const char* resume_file);
  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You should implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk together with the learned net.
  void Snapshot() {
    SnapshotWithScores(vector<float>());
  }
  void SnapshotWithScores(const vector<float>& scores);
  virtual ~Solver();

  const SolverParameter& param() const {
    return param_;
  }
  shared_ptr<Net> net() {
    if (net_) {
      net_->set_solver(this);
    }
    return net_;
  }
  const vector<shared_ptr<Net>>& test_nets() {
    return test_nets_;
  }
  int iter() const {
    return iter_;
  }
  int relative_iter() const {
    return iter_ - iterations_restored_;
  }
  float total_lapse() const {
    return total_lapse_;
  }
  bool is_root() const {
    return root_solver_ == nullptr;
  }
  int rank() const {
    return rank_;
  }

  float perf_report(std::ostream& os, int device, int align = 0) const;

  // Invoked at specific points during an iteration
  class Callback {
   public:
    virtual void allreduce(int param_id) = 0;
    virtual void allreduce_bucket(size_t count, void* bucket, Type type) = 0;
    virtual void soft_barrier(int b) = 0;
    virtual void cancel_all() = 0;
    virtual void saveTestResults(float loss, const vector<float>& scores) = 0;
    virtual void aggregateTestResults(float* loss, vector<float>* scores) = 0;
    virtual cudaStream_t comm_stream() const = 0;

   protected:
    virtual void on_start(const vector<shared_ptr<Blob>>& net) = 0;
    friend class Solver;
  };

  Callback* callback() const {
    return callback_;
  }
  void set_callback(Callback* value) {
    callback_ = value;
  }
  void root_add_callback(Callback* value) {
    root_callbacks_.push_back(value);
  }

  Flag iter_flag_;

  void iteration_complete_signal() {
    iter_flag_.set();
  }
  void iteration_start_signal() {
    iter_flag_.reset();
  }
  void iteration_wait() {
    iter_flag_.wait();
  }
  void iteration_cancel() {
    iter_flag_.disarm();
  }
  void stop_reducing() const {
    net_->Finalize();
  }

  void CheckSnapshotWritePermissions();

  void request_early_exit() {
    requested_early_exit_ = true;
    iteration_complete_signal();
  }

  bool display() const {
    return param_.display() && iter_ % param_.display() == 0;
  }

  bool param_display() const {
    return param_.display() > 0;
  }

  bool initialized() const {
    return init_flag_.is_set();
  }

  Type data_type() const {
    return data_type_;
  }

  /**
   * @brief Returns the solver type.
   */
  virtual const char* type() const { return ""; }
  virtual void PrintRate(float rate = 0) {}
  virtual float GetLearningRate() const = 0;
  virtual void ClipGradientsAndNormalize(void* handle, int type_id,
      const std::set<int>& param_ids) = 0;
  virtual void ApplyUpdate(int param_id, void* handle, float rate, bool normalize,
      bool clear_grads) = 0;

 protected:
  string SnapshotFilename(const string& extension, const vector<float>& scores) const;
  string SnapshotToBinaryProto(const vector<float>& scores);
  string SnapshotToHDF5(const vector<float>& scores);
  // The test routine
  vector<float> TestAll(const int iters = 0, bool use_multi_gpu = false);
  vector<float> Test(const int test_net_id = 0, const int iters = 0, bool use_multi_gpu = false);
  vector<float> TestDetection(const int test_net_id = 0, const int iters = 0);
  virtual void SnapshotSolverState(const string& model_filename) = 0;
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
  void UpdateSmoothedLoss(float loss, int start_iter, int average_loss);
  void Reduce(Callback* callback, int device, Caffe::Brew mode, uint64_t rand_seed,
              bool root_solver);

  void callback_soft_barrier() {
    if (callback_ != nullptr) {
      callback_->soft_barrier(0);
    }
  }

  const SolverParameter param_;
  const Type data_type_;
  int iter_;
  int id_;
  float total_lapse_;
  mutable int current_step_;
  shared_ptr<Net> net_;
  vector<shared_ptr<Net>> test_nets_;
  Callback* callback_;
  vector<Callback*> root_callbacks_;
  vector<float> losses_;
  float smoothed_loss_;

  // The root solver that holds root nets (actually containing shared layers)
  // in data parallelism
  const Solver* const root_solver_;
  const size_t rank_;

  // A function that can be set by a client of the Solver to provide indication
  // that it wants a snapshot saved and/or to exit early.
  ActionCallback action_request_function_;

  // True iff a request to stop early was received.
  bool requested_early_exit_;

  // some layers like Data have to wait for this one
  Flag init_flag_;

  // Timing information
  shared_ptr<Timer> iteration_timer_;
  shared_ptr<Timer> test_timer_;
  int iterations_last_;
  int iterations_restored_;

  static constexpr size_t MAX_SNAPSHOT_SCORES = 5;

  DISABLE_COPY_MOVE_AND_ASSIGN(Solver);
};

}  // namespace caffe

#endif  // CAFFE_SOLVER_HPP_
