import os
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
from warnings import filterwarnings

import numpy as np
import optuna
from numpy import ndarray
from optuna import Trial
from pandas import DataFrame
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import BaseCrossValidator, LeaveOneOut, StratifiedShuffleSplit
from sklearn.model_selection import cross_validate as cv
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DTreeClassifier
from typing_extensions import Literal

Splits = Iterable[Tuple[ndarray, ndarray]]
Classifier = Literal["rf", "svm", "dtree", "mlp", "bag", "lda"]
Kernel = Literal["rbf", "linear", "sigmoid"]
ValMethod = Literal["holdout", "kfold", "k-fold", "loocv", "mc", "none"]
CVMethod = Union[int, float, Literal["loocv", "mc"]]
MultiTestCVMethod = Union[int, Literal["mc"]]


LR_SOLVER = "liblinear"
# MLP_LAYER_SIZES = [0, 8, 32, 64, 128, 256, 512]
MLP_LAYER_SIZES = [4, 8, 16, 32]
N_SPLITS = 5
TEST_SCORES = ["accuracy", "roc_auc"]
SEED = 3
VAL_SIZE = 0.2

RF_BASE_ARGS = dict(n_jobs=-1, n_estimators=1000, bootstrap=True, max_depth=None)
SVM_BASE_ARGS = dict(cache_size=500)
DTREE_BASE_ARGS = dict()
MLP_BASE_ARGS = dict()
LDA_BASE_ARGS = dict()
BASE_ARGS: Dict[Classifier, Dict] = {
    "rf": RF_BASE_ARGS,
    "svm": SVM_BASE_ARGS,
    "dtree": DTREE_BASE_ARGS,
    "mlp": MLP_BASE_ARGS,
    "bag": dict(),
    "lda": LDA_BASE_ARGS,
}


def bagger(**kwargs: Any) -> Callable:
    """Helper for uniform interface only"""
    return BaggingClassifier(base_estimator=LR(solver=LR_SOLVER), **kwargs)  # type: ignore


CLASSIFIERS: Dict[str, Callable] = {
    "rf": RF,
    "svm": SVC,
    "dtree": DTreeClassifier,
    "mlp": MLP,
    "bag": bagger,
    "lda": LDA,
}


def get_classifier_constructor(name: Classifier) -> Tuple[Callable, Dict]:
    constructor = CLASSIFIERS[name]
    return constructor, BASE_ARGS[name]


@dataclass(init=True, repr=True, eq=True, frozen=True)
class HtuneResult:
    classifier: Classifier
    n_trials: int
    cv_method: CVMethod
    val_acc: float = np.nan
    best_params: Dict = field(default_factory=dict)


def train_val_splits(
    df: DataFrame, val_size: float = VAL_SIZE
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """Wrapper around `sklearn.model_selection.train_test_split` to return splits as `DataFrame`s
    instead of numpy arrays.

    Parameters
    ----------
    df: DataFrame
        Data with target in column named "target".

    val_size: float = 0.2
        Percent of data to reserve for validation

    Returns
    -------
    splits: [X_train, X_val, y_train, y_val]
    """
    train, val = train_test_split(
        df, test_size=val_size, random_state=SEED, shuffle=True, stratify=df.target
    )
    X_train = train.drop(columns="target")
    X_val = val.drop(columns="target")
    y_train = train.target.astype(int)
    y_val = val.target.astype(int)
    return X_train, X_val, y_train, y_val


def get_cv(y_train: DataFrame, cv_method: CVMethod) -> Union[int, Splits, BaseCrossValidator]:
    """Helper to construct an object that `sklearn.model_selection.cross_validate` will accept in
    its `cv` argument

    Parameters
    ----------
    y_train: DataFrame
        Needed for stratification.

    cv_method: CVMethod
        Method to create object for.

    Returns
    -------
    cv: Union[int, Splits, BaseCrossValidator]
        The object that can be passed into the `cross_validate` function
    """
    if isinstance(cv_method, int):
        return int(cv_method)
    if isinstance(cv_method, float):  # stratified holdout
        if cv_method <= 0 or cv_method >= 1:
            raise ValueError("Holdout test_size must be in (0, 1)")
        test_size = cv_method
        y = np.array(y_train).ravel()
        idx = np.arange(y.shape[0])
        return [
            train_test_split(idx, test_size=test_size, random_state=SEED, shuffle=True, stratify=y)
        ]
    cv_method = str(cv_method).lower()  # type: ignore
    if cv_method == "loocv":
        return LeaveOneOut()
    if cv_method == "mc":
        return StratifiedShuffleSplit(n_splits=20, test_size=0.2, random_state=SEED)
    raise ValueError("Invalid `cv_method`")


def cv_desc(cv_method: CVMethod) -> str:
    """Helper for logging a readable description of the CVMethod to stdout"""
    if isinstance(cv_method, int):
        return f"stratified {cv_method}-fold"
    if isinstance(cv_method, float):  # stratified holdout
        if cv_method <= 0 or cv_method >= 1:
            raise ValueError("Holdout test_size must be in (0, 1)")
        perc = int(100 * cv_method)
        return f"stratified {perc}% holdout"
    cv_method = str(cv_method).lower()  # type: ignore
    if cv_method == "loocv":
        return "LOOCV"
    if cv_method == "mc":
        return "stratified Monte-Carlo (20 random 20%-sized test sets)"
    raise ValueError("Invalid `cv_method`")


"""See Optuna docs (https://optuna.org/#code_ScikitLearn) for the motivation behond the closures
below. Currently I am using closures, but this might be a BAD IDEA in parallel contexts. In any
case, they do seem to suggest this is OK https://optuna.readthedocs.io/en/stable/faq.html
#how-to-define-objective-functions-that-have-own-arguments, albeit by using classes or lambdas. """


def svm_objective(
    X_train: DataFrame, y_train: DataFrame, cv_method: CVMethod = 5
) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        args: Dict = dict(
            kernel=trial.suggest_categorical("kernel", choices=["rbf"]),
            C=trial.suggest_loguniform("C", 1e-10, 1e10),
        )
        _cv = get_cv(y_train, cv_method)
        estimator = SVC(**{**SVM_BASE_ARGS, **args})
        scores = cv(estimator, X=X_train, y=y_train, scoring="accuracy", cv=_cv, n_jobs=-1)
        return float(np.mean(scores["test_score"]))

    return objective


def lda_objective(
    X_train: DataFrame, y_train: DataFrame, cv_method: CVMethod = 5
) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        args: Dict = dict(
            # kernel=trial.suggest_categorical("kernel", choices=["rbf"]),
            # C=trial.suggest_loguniform("C", 1e-10, 1e10),
        )
        _cv = get_cv(y_train, cv_method)
        estimator = LDA(**args)
        scores = cv(estimator, X=X_train, y=y_train, scoring="accuracy", cv=_cv, n_jobs=-1)
        return float(np.mean(scores["test_score"]))

    return objective


def rf_objective(
    X_train: DataFrame, y_train: DataFrame, cv_method: CVMethod = 5
) -> Callable[[Trial], float]:
    """We follow papers and guides here https://stats.stackexchange.com/a/346984

    As per https://arxiv.org/pdf/1705.05654.pdf, there is no point tuning n_estimators: more will
    almost always be better, 500-1000 if computationally feasible. The case is likely similar for
    max_depth, and so should be left at None. Tuning max_samples is reasonable with bootstrap=True,
    as is tuning max_features.
    """

    def objective(trial: Trial) -> float:
        args: Dict = dict(
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
            # bootstrap=trial.suggest_categorical("bootstrap", [True]),
            max_features=trial.suggest_uniform("max_features", 0.1, 1.0),
            max_samples=trial.suggest_uniform("max_samples", 0.1, 1.0),
        )
        _cv = get_cv(y_train, cv_method)
        estimator = RF(**{**RF_BASE_ARGS, **args})
        scores = cv(estimator, X=X_train, y=y_train, scoring="accuracy", cv=_cv, n_jobs=1)
        return float(np.mean(scores["test_score"]))

    return objective


def dtree_objective(
    X_train: DataFrame, y_train: DataFrame, cv_method: CVMethod = 5
) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        args: Dict = dict(
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
            splitter=trial.suggest_categorical("splitter", ["best", "random"]),
            max_depth=trial.suggest_int("max_depth", 2, 50),
        )
        _cv = get_cv(y_train, cv_method)
        estimator = DTreeClassifier(**{**DTREE_BASE_ARGS, **args})
        scores = cv(estimator, X=X_train, y=y_train, scoring="accuracy", cv=_cv, n_jobs=-1)
        return float(np.mean(scores["test_score"]))

    return objective


def logistic_bagging_objective(
    X_train: DataFrame, y_train: DataFrame, cv_method: CVMethod = 5
) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        args: Dict = dict(
            n_estimators=trial.suggest_int("n_estimators", 5, 23, 2),
            max_features=trial.suggest_uniform("max_features", 0, 1),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
        )
        _cv = get_cv(y_train, cv_method)
        estimator = BaggingClassifier(
            base_estimator=LR(solver=LR_SOLVER), random_state=SEED, n_jobs=2, **args
        )
        scores = cv(estimator, X=X_train, y=y_train, scoring="accuracy", cv=_cv, n_jobs=4)
        return float(np.mean(scores["test_score"]))

    return objective


def mlp_layers(l1: int, l2: int, l3: int, l4: int, l5: int) -> Tuple[int, ...]:
    """
    Needed for converting randomly generated layer sizes into an argument the MLP classifier can
    understand. I strongly suspect this method mucks up Optuna's Bayesian optimization though if we
    allow and layer to have a size zero, since this would effectively be changing the meaning of
    that hyperparam.
    """
    layers = [l1, l2, l3, l4, l5]
    return tuple([layer for layer in layers if layer > 0])


def mlp_args_from_params(params: Dict) -> Dict:
    """Convert the params returned from trial.best_params into a form that can be used by
    MLPClassifier"""
    d = {**params}
    l1 = d.pop("l1")
    l2 = d.pop("l2")
    l3 = d.pop("l3")
    l4 = d.pop("l4")
    l5 = d.pop("l5")
    d["hidden_layer_sizes"] = mlp_layers(l1, l2, l3, l4, l5)
    return d


def mlp_objective(
    X_train: DataFrame, y_train: DataFrame, cv_method: CVMethod = 5
) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        l1 = trial.suggest_categorical("l1", choices=MLP_LAYER_SIZES)
        l2 = trial.suggest_categorical("l2", choices=MLP_LAYER_SIZES)
        l3 = trial.suggest_categorical("l3", choices=MLP_LAYER_SIZES)
        l4 = trial.suggest_categorical("l4", choices=MLP_LAYER_SIZES)
        l5 = trial.suggest_categorical("l5", choices=MLP_LAYER_SIZES)
        args: Dict = dict(
            hidden_layer_sizes=mlp_layers(l1, l2, l3, l4, l5),
            activation=trial.suggest_categorical("activation", ["relu"]),
            solver=trial.suggest_categorical("solver", ["adam"]),
            # alpha=trial.suggest_loguniform("alpha", 1e-8, 1e-1),
            alpha=trial.suggest_loguniform("alpha", 1e-7, 1e-2),
            batch_size=trial.suggest_categorical("batch_size", choices=[8, 16, 32]),
            learning_rate=trial.suggest_categorical(
                "learning_rate", choices=["constant", "adaptive"]
            ),
            learning_rate_init=trial.suggest_loguniform("learning_rate_init", 5e-5, 5e-2),
            max_iter=trial.suggest_categorical("max_iter", [100]),
            early_stopping=trial.suggest_categorical("early_stopping", [False]),
            validation_fraction=trial.suggest_categorical("validation_fraction", [0.1]),
        )
        mlp = MLP(**args)
        # https://stackoverflow.com/questions/53784971/how-to-disable-convergencewarning-using-sklearn
        before = os.environ.get("PYTHONWARNINGS", "")
        os.environ["PYTHONWARNINGS"] = "ignore"  # can't kill ConvergenceWarning any other way
        filterwarnings("ignore", category=ConvergenceWarning)
        _cv = get_cv(y_train, cv_method)
        scores = cv(mlp, X=X_train, y=y_train, scoring="accuracy", cv=_cv, n_jobs=-1)
        os.environ["PYTHONWARNINGS"] = before
        acc = float(np.mean(scores["test_score"]))
        return acc

    return objective


def hypertune_classifier(
    classifier: Classifier,
    X_train: DataFrame,
    y_train: DataFrame,
    n_trials: int = 200,
    cv_method: CVMethod = 5,
    verbosity: int = optuna.logging.ERROR,
) -> HtuneResult:
    """Core function. Uses Optuna base TPESampler (Tree-Parzen Estimator Sampler) to perform
    Bayesian hyperparameter optimization via Gaussian processes on the classifier specified in
    `classifier`.

    Parameters
    ----------
    classifier: Classifier
        Classifier to tune.

    X_train: DataFrame
        DataFrame with no target value (features only). Shape (n_samples, n_features)

    y_train: DataFrame
        Target values. Shape (n_samples,).

    n_trials: int = 200
        Number of trials to use with Optuna.

    cv_method: CVMethod = 5
        How to evaluate accuracy during tuning.

    verbosity: int = optuna.logging.ERROR
        See https://optuna.readthedocs.io/en/stable/reference/logging.html. Most useful other option
        is `optuna.logging.INFO`.

    Returns
    -------
    htuned: HtuneResult
        See top of this file.
    """
    OBJECTIVES: Dict[str, Callable] = {
        "rf": rf_objective(X_train, y_train, cv_method),
        "svm": svm_objective(X_train, y_train, cv_method),
        "dtree": dtree_objective(X_train, y_train, cv_method),
        "mlp": mlp_objective(X_train, y_train, cv_method),
        "bag": logistic_bagging_objective(X_train, y_train, cv_method),
        "lda": lda_objective(X_train, y_train, cv_method),
    }
    # HYPERTUNING
    objective = OBJECTIVES[classifier]
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    optuna.logging.set_verbosity(verbosity)
    if classifier == "mlp":
        # https://stackoverflow.com/questions/53784971/how-to-disable-convergencewarning-using-sklearn
        before = os.environ.get("PYTHONWARNINGS", "")
        os.environ["PYTHONWARNINGS"] = "ignore"  # can't kill ConvergenceWarning any other way

    study.optimize(objective, n_trials=n_trials)

    if classifier == "mlp":
        os.environ["PYTHONWARNINGS"] = before

    val_method = cv_desc(cv_method)
    acc = np.round(study.best_value, 3)
    if verbosity != optuna.logging.ERROR:
        print(f"\n{' Tuning Results ':=^80}")
        print("Best params:")
        pprint(study.best_params, indent=4, width=80)
        print(f"\nTuning validation: {val_method}")
        print(f"Best accuracy:      μ = {acc:0.3f}")
        # print("=" * 80, end="\n")

    return HtuneResult(
        classifier=classifier,
        n_trials=n_trials,
        cv_method=cv_method,
        val_acc=study.best_value,
        best_params=study.best_params,
    )


def evaluate_hypertuned(
    htuned: HtuneResult,
    cv_method: CVMethod,
    X_train: DataFrame,
    y_train: DataFrame,
    X_test: Optional[DataFrame] = None,
    y_test: Optional[DataFrame] = None,
    log: bool = True,
) -> Dict[str, Any]:
    """Core function. Given teh result of hypertuning, evaluate the final parameters.

    Parameters
    ----------
    htuned: HtuneResult
        Results from `src.hypertune.hypertune_classifier`.

    cv_method: CVMethod = 5
        How to evaluate accuracy during tuning.

    X_train: DataFrame
        DataFrame with no target value (features only). Shape (n_samples, n_features)

    y_train: DataFrame
        Target values. Shape (n_samples,).

    X_test: DataFrame = None
        DataFrame with no target value (features only). Shape (n_test_samples, n_features), if using
        a test sample held out during hyperparameter tuning.

    y_test: DataFrame = None
        Target values. Shape (n_test_samples,), corresponding to X_test.

    log: bool = True
        If True, print results to console.

    Returns
    -------
    result: Dict[str, Any]
        A dict with structure:

            {
                htuned: HtuneResult,
                cv_method: CVMethod,  # The method used during hypertuning
                acc: float  # mean accuracy across folds
                auc: float  # mean AUC across folds
                acc_sd: float  # sd of accuracy across folds
                auc_sd: float  # sd of AUC across folds
            }
    """
    classifier_name = htuned.classifier
    params = htuned.best_params
    args = mlp_args_from_params(params) if classifier_name == "mlp" else params
    constructor, base_args = get_classifier_constructor(classifier_name)
    estimator = constructor(**{**base_args, **args})
    if (X_test is None) and (y_test is None):
        _cv = get_cv(y_train, cv_method)
        scores = cv(estimator, X=X_train, y=y_train, scoring=TEST_SCORES, cv=_cv)
        acc_mean = float(np.mean(scores["test_accuracy"]))
        auc_mean = float(np.mean(scores["test_roc_auc"]))
        acc_sd = float(np.std(scores["test_accuracy"], ddof=1))
        auc_sd = float(np.std(scores["test_roc_auc"], ddof=1))
        desc = cv_desc(cv_method)
        result = dict(
            htuned=htuned,
            cv_method=htuned.cv_method,
            acc=np.mean(scores["test_accuracy"]),
            auc=np.mean(scores["test_roc_auc"]),
            acc_sd=np.std(scores["test_accuracy"], ddof=1),
            auc_sd=np.std(scores["test_roc_auc"], ddof=1),
        )
        if not log:
            return result
        # fmt: off
        print(f"Testing validation: {desc}")
        print(f"Accuracy:           μ = {np.round(acc_mean, 3):0.3f} (sd = {np.round(acc_sd, 4):0.4f})")  # noqa
        print(f"AUC:                μ = {np.round(auc_mean, 3):0.3f} (sd = {np.round(auc_sd, 4):0.4f})")  # noqa
        # fmt: on
        return result
    elif (X_test is not None) and (y_test is not None):
        y_pred = estimator.fit(X_train, y_train).predict(X_test)
        y_score = (
            estimator.decision_function(X_test)
            if classifier != "mlp"
            else estimator.predict_proba(X_test)
        )
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_score)
        percent = int(100 * float(cv_method))
        scores = dict(test_accuracy=np.array([acc]).ravel(), test_roc_auc=np.array([auc]).ravel())
        result = dict(
            htuned=htuned,
            cv_method=cv_method,
            acc=np.mean(scores["test_accuracy"]),
            auc=np.mean(scores["test_roc_auc"]),
            acc_sd=np.nan,
            auc_sd=np.nan,
        )
        if not log:
            return result
        print(f"Testing validation: {percent}% holdout")
        print(f"          Accuracy: μ = {np.round(acc, 3):0.3f} (sd = {np.round(acc, 4):0.4f})")
        print(f"               AUC: μ = {np.round(auc, 3):0.3f} (sd = {np.round(auc, 4):0.4f})")
        return result
    else:
        raise ValueError("Invalid test data: only one of `X_test` or `y_test` was None.")


def evaluate_untuned(
    classifier: Classifier,
    cv_method: CVMethod,
    X_train: DataFrame,
    y_train: DataFrame,
    X_test: Optional[DataFrame] = None,
    y_test: Optional[DataFrame] = None,
    log: bool = True,
) -> Dict[str, Any]:
    htuned = HtuneResult(
        classifier=classifier,
        n_trials=-1,
        cv_method=cv_method,
        val_acc=-1,
        best_params=BASE_ARGS[classifier],
    )
    return evaluate_hypertuned(htuned, cv_method, X_train, y_train, X_test, y_test, log)
