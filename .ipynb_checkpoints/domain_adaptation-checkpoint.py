"""
Модуль для проведения экспериментов по доменной адаптации
данных космических аппаратов ACE и DSCOVR при прогнозировании Dst-индекса.

Содержит:
1. Функции разбиения данных по сценариям (без адаптации)
2. Функции обучения моделей адаптации (DSCOVR->ACE, ACE->DSCOVR, cross-domain)
3. Универсальную функцию оценки прогностических моделей с усреднением по сидам
4. Функции для запуска сценариев
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

warnings.filterwarnings('ignore')

# Модели, считающиеся стохастическими (требуют усреднения по сидам)
STOCHASTIC_MODELS = {'LGBM', 'MLP'}


# =====================================================================
# 1. РАЗБИЕНИЕ ДАННЫХ ПО СЦЕНАРИЯМ БЕЗ АДАПТАЦИИ
# =====================================================================

def split_datasets_by_scenario(ace, disc, split_date, scenario):
    """
    Разделяет два датасета ace и discover на train/test по заданной дате split_date.
    При необходимости выравнивает временные индексы (пересечение) и удаляет строки с NaN.

    Сценарии:
        'ace-ace'   — train/test на ACE
        'ace-disc'  — train на ACE, test на DSCOVR
        'disc-ace'  — train на DSCOVR (overlap), test на ACE
        'disc-disc' — train/test на DSCOVR (overlap)
    """
    a, d = ace.copy(), disc.copy()
    split_date = pd.Timestamp(split_date)

    if scenario == 'ace-ace':
        train = a[a.index < split_date]
        test = a[a.index >= split_date]

    elif scenario == 'ace-disc':
        train = a[a.index < split_date]
        test = d[d.index >= split_date]

    elif scenario == 'disc-ace':
        common_idx = a.index.intersection(d.index)
        a_aligned = a.loc[common_idx].dropna(how='any')
        d_aligned = d.loc[common_idx].dropna(how='any')
        train = d_aligned[d_aligned.index < split_date]
        test = a_aligned[a_aligned.index >= split_date]

    elif scenario == 'disc-disc':
        common_idx = a.index.intersection(d.index)
        d_aligned = d.loc[common_idx].dropna(how='any')
        train = d_aligned[d_aligned.index < split_date]
        test = d_aligned[d_aligned.index >= split_date]

    else:
        raise ValueError(f"Неизвестный сценарий: '{scenario}'")

    return train.copy(), test.copy()


def make_combined_train(ace_df, disc_df, split_date, add_is_ace_flag=True):
    """
    Подготовка данных для shuffle-сценария:
    train = ACE_train + DSCOVR_train (с флагом is_ace),
    test  = DSCOVR_test.
    """
    ace = ace_df.sort_index().copy()
    disc = disc_df.sort_index().copy()
    split_date = pd.Timestamp(split_date)

    if add_is_ace_flag:
        ace['is_ace'] = 1
        disc['is_ace'] = 0

    ace_train = ace.loc[:split_date]
    disc_train = disc.loc[:split_date]
    disc_test = disc.loc[split_date:]

    combined_train = pd.concat([ace_train, disc_train]).sort_index()
    return combined_train, disc_test


# =====================================================================
# 2. МОДЕЛИ АДАПТАЦИИ
# =====================================================================

def adapt_dscovr_to_ace(ace_df, disc_df, future_lags, split_date,
                        adapt_model, multi_output=False):
    """
    Сценарий DSCOVR -> ACE

    Модель L: DSCOVR -> ACE обучается на пересечении тренировочного набора,
    затем адаптирует тестовый набор DSCOVR в домен ACE

    Параметры
    ---------
    ace_df, disc_df : pd.DataFrame
        Исходные данные с индексом datetime
    future_lags : list[str]
        Колонки целевых лагов Dst
    split_date : str
        Дата разбиения на train/test
    adapt_model : необученная модель
    multi_output : bool
        Обернуть ли модель в MultiOutputRegressor (для деревьев)

    Возвращает
    ----------
    disc_test_adapted : pd.DataFrame
        Адаптированный тестовый DSCOVR в домене ACE
    artifacts : dict
        {'model': L, 'scaler_src': sc_disc, 'scaler_tgt': sc_ace}
    """
    ace = ace_df.sort_index()
    disc = disc_df.sort_index()
    split_date = pd.Timestamp(split_date)

    ace_train = ace.loc[:split_date]
    disc_train = disc.loc[:split_date]
    disc_test = disc.loc[split_date:]

    overlap_idx = ace_train.index.intersection(disc_train.index)
    ace_overlap = ace_train.loc[overlap_idx]
    disc_overlap = disc_train.loc[overlap_idx]

    feature_cols = [c for c in ace.columns if c not in future_lags]

    sc_ace = StandardScaler().fit(ace_overlap[feature_cols])
    sc_disc = StandardScaler().fit(disc_overlap[feature_cols])

    X_disc = sc_disc.transform(disc_overlap[feature_cols])
    X_ace = sc_ace.transform(ace_overlap[feature_cols])

    L = MultiOutputRegressor(clone(adapt_model)) if multi_output else clone(adapt_model)
    L.fit(X_disc, X_ace)

    # Применяем к тесту DSCOVR
    X_disc_test = sc_disc.transform(disc_test[feature_cols])
    X_disc_test_adapted = sc_ace.inverse_transform(L.predict(X_disc_test))

    disc_test_adapted = pd.DataFrame(
        X_disc_test_adapted, index=disc_test.index, columns=feature_cols
    )
    # Целевые лаги переносим без изменений
    for col in future_lags:
        if col in disc_test.columns:
            disc_test_adapted[col] = disc_test[col]

    artifacts = {'model': L, 'scaler_src': sc_disc, 'scaler_tgt': sc_ace}
    return disc_test_adapted, artifacts


def adapt_ace_to_dscovr(ace_df, disc_df, future_lags, split_date,
                        adapt_model, multi_output=False):
    """
    Сценарий ACE -> DSCOVR

    Модель K: ACE -> DSCOVR обучается на пересечении тренировочного набора,
    затем адаптирует исторические данные ACE (до появления DSCOVR) в домен DSCOVR

    Возвращает
    ----------
    ace_old_adapted : pd.DataFrame
        Адаптированные старые данные ACE в домене DSCOVR
    disc_train : pd.DataFrame
        Исходный train DSCOVR
    disc_test : pd.DataFrame
        Исходный test DSCOVR
    artifacts : dict
    """
    ace = ace_df.sort_index()
    disc = disc_df.sort_index()
    split_date = pd.Timestamp(split_date)

    ace_train = ace.loc[:split_date]
    disc_train = disc.loc[:split_date]
    disc_test = disc.loc[split_date:]

    overlap_idx = ace_train.index.intersection(disc_train.index)
    ace_overlap = ace_train.loc[overlap_idx]
    disc_overlap = disc_train.loc[overlap_idx]

    feature_cols = [c for c in ace.columns if c not in future_lags]

    sc_ace = StandardScaler().fit(ace_overlap[feature_cols])
    sc_disc = StandardScaler().fit(disc_overlap[feature_cols])

    X_ace = sc_ace.transform(ace_overlap[feature_cols])
    X_disc = sc_disc.transform(disc_overlap[feature_cols])

    K = MultiOutputRegressor(clone(adapt_model)) if multi_output else clone(adapt_model)
    K.fit(X_ace, X_disc)

    # Адаптируем исторический ACE (до появления DSCOVR)
    ace_old = ace.loc[:disc.index.min()][feature_cols]
    X_ace_old = sc_ace.transform(ace_old)
    X_ace_old_adapted = sc_disc.inverse_transform(K.predict(X_ace_old))

    ace_old_adapted = pd.DataFrame(
        X_ace_old_adapted, index=ace_old.index, columns=feature_cols
    )
    for col in future_lags:
        if col in ace_df.columns:
            ace_old_adapted[col] = ace_df.loc[ace_old.index, col]

    artifacts = {'model': K, 'scaler_src': sc_ace, 'scaler_tgt': sc_disc}
    return ace_old_adapted, disc_train, disc_test, artifacts


def cross_adapt_domain(src_df, tgt_df, future_lags,
                       field_model_factory,
                       wind_model_factory,
                       field_regex=r'^(Dst|bx_gsm|by_gsm|bz_gsm|bt)(_lag\d+)?$',
                       wind_regex=r'^(proton_density|proton_speed|proton_temperature)(_lag\d+)?$',
                       direction='disc-to-ace',
                       split_date=None):
    """
    Комбинированная адаптация:
      - параметры ММП: отдельная модель для каждой колонки (one-to-one)
      - параметры солнечного ветра: модель many-to-one на всём наборе переменных ветра

    Параметры
    ---------
    src_df, tgt_df : pd.DataFrame
        src — исходный домен (откуда переводим), tgt — целевой
    field_model_factory : callable -> sklearn-Pipeline
        Пайплайн для адаптации полей ММП
    wind_model_factory : callable -> sklearn-Pipeline
        Пайплайн для адаптации параметров СВ.
    direction : {'disc-to-ace', 'ace-to-disc'}
        Сценарий адаптации
    split_date : str | pd.Timestamp | None
        Если задано — обучение на overlap до split_date, адаптация — на нужном куске
        Если None — обучение и адаптация на всём пересечении

    Возвращает
    ----------
    adapted_df : pd.DataFrame
        Адаптированный набор данных в целевом домене tgt
    """
    src = src_df.sort_index()
    tgt = tgt_df.sort_index()

    field_features = [c for c in src.columns
                      if c in tgt.columns and pd.Series([c]).str.match(field_regex)[0]]
    wind_features = [c for c in src.columns
                     if c in tgt.columns and pd.Series([c]).str.match(wind_regex)[0]]

    # Определяем пересечение данных на train (overlap) и часть для адаптации
    overlap_idx = src.index.intersection(tgt.index)
    if split_date is not None:
        split_date = pd.Timestamp(split_date)
        train_idx = overlap_idx[overlap_idx < split_date]
    else:
        train_idx = overlap_idx

    src_train = src.loc[train_idx]
    tgt_train = tgt.loc[train_idx]

    # что адаптируем
    if direction == 'disc-to-ace':
        # адаптируем тест DSCOVR (после split_date)
        if split_date is None:
            to_adapt = src.copy()
        else:
            to_adapt = src.loc[src.index >= split_date].copy()
    elif direction == 'ace-to-disc':
        # адаптируем старый ACE (до начала DSCOVR)
        to_adapt = src.loc[:tgt.index.min()].copy()
    else:
        raise ValueError(f"Неизвестное direction: {direction}")

    adapted = to_adapt.copy()

    # Адаптация полей: one-to-one
    for col in field_features:
        m = field_model_factory()
        m.fit(src_train[[col]], tgt_train[col])
        adapted[col] = m.predict(to_adapt[[col]])

    # Адаптация ветра: many-to-one
    if wind_features:
        for target_col in wind_features:
            m = wind_model_factory()
            m.fit(src_train[wind_features], tgt_train[target_col])
            adapted[target_col] = m.predict(to_adapt[wind_features])

    # Целевые лаги — без изменений
    for col in future_lags:
        if col in src_df.columns:
            adapted[col] = src_df.loc[adapted.index, col]

    return adapted


# =====================================================================
# 3. ОЦЕНКА ПРОГНОСТИЧЕСКИХ МОДЕЛЕЙ
# =====================================================================

def _fit_predict_one_model(name, model, X_train, y_train, X_test, random_seed):
    """
    Обучение одной модели + предсказание. Возвращает y_pred
    """
    # Экспериментально подобрано, что без валидационного набора метрики лучше
    # if name == 'LGBM':
    #     X_tr, X_val, y_tr, y_val = train_test_split(
    #         X_train, y_train, test_size=0.1, random_state=random_seed
    #     )
    #     model.fit(
    #         X_tr, y_tr,
    #         boost__eval_set=[(X_val, y_val)],
    #         boost__eval_metric='l2'
    #     )
    # else:
    #     model.fit(X_train, y_train)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def _compute_metrics(y_true, y_pred):
    return {
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE':  float(mean_absolute_error(y_true, y_pred)),
        'R2':   float(r2_score(y_true, y_pred)),
    }


def evaluate_models_one_seed(train_df, test_df, target, future_lags,
                             build_models_fn, random_seed,
                             scenario_label, delays):
    """
    Один прогон всех моделей, заданных с помощью build_models_fn(random_state),
    Возвращает список словарей с метриками
    """
    feature_cols = [c for c in train_df.columns if c not in future_lags]
    train = train_df.dropna(subset=feature_cols + [target])
    test = test_df.dropna(subset=feature_cols + [target])

    X_train, y_train = train[feature_cols].values, train[target].values
    X_test, y_test = test[feature_cols].values, test[target].values

    models = build_models_fn(random_state=random_seed)
    results = []

    for name, model in models.items():
        try:
            y_pred = _fit_predict_one_model(
                name, model, X_train, y_train, X_test, random_seed
            )
            metrics = _compute_metrics(y_test, y_pred)
            results.append({
                'target': target,
                'delays': delays,
                'scenario': scenario_label,
                'forecast_model': name,
                'seed': random_seed,
                **metrics,
            })
        except Exception as e:
            print(f"[{scenario_label}] {name} seed={random_seed} -> Ошибка: {e}")

    return results


def evaluate_with_seeds(train_df, test_df, target, future_lags,
                        build_models_fn,
                        scenario_label, delays='24h',
                        n_seeds=3, base_seed=42,
                        results_list=None, verbose=True):
    """
    Универсальная функция оценки моделей с усреднением по сидам.
    
    Линейные модели обучаются один раз (детерминированы),
    стохастические — n_seeds раз

    Параметры
    ---------
    train_df, test_df : pd.DataFrame
    target : str
        Колонка целевой переменной (например 'Dst_plus1')
    future_lags : list[str]
        Колонки всех погруженных переменных таргета (исключаются из признаков)
    build_models_fn : callable(random_state) -> dict[name, Pipeline]
        Прогностические модели
    scenario_label : str
        Метка сценария для записи в результаты
    n_seeds : int
        Сколько раз запускать стохастические модели
    results_list : list | None
        Список для сбора результатов

    Возвращает
    ----------
    df_all : pd.DataFrame
        Сырые результаты по каждому сиду
    df_agg : pd.DataFrame
        Агрегированные (mean ± std)
    results_list : list
        Обновлённый список результатов
    """
    if results_list is None:
        results_list = []

    # Один прогон — для всех моделей (включая линейные)
    single_run = evaluate_models_one_seed(
        train_df, test_df, target, future_lags,
        build_models_fn, random_seed=base_seed,
        scenario_label=scenario_label, delays=delays,
    )
    df_single = pd.DataFrame(single_run)

    # Дополнительные сиды только для стохастических
    seeds = [base_seed + i for i in range(n_seeds)]
    all_stoch = []
    for seed in seeds:
        run = evaluate_models_one_seed(
            train_df, test_df, target, future_lags,
            build_models_fn, random_seed=seed,
            scenario_label=scenario_label, delays=delays,
        )
        all_stoch.extend([r for r in run if r['forecast_model'] in STOCHASTIC_MODELS])
    df_stoch = pd.DataFrame(all_stoch)

    # Агрегация
    records = []

    # Детерминированные (линейные) — берем из single_run
    df_det = df_single[~df_single['forecast_model'].isin(STOCHASTIC_MODELS)]
    for _, row in df_det.iterrows():
        records.append({
            'target': row['target'],
            'delays': delays,
            'scenario': scenario_label,
            'forecast_model': row['forecast_model'],
            'RMSE': row['RMSE'], 'RMSE_std': 0.0,
            'MAE': row['MAE'],   'MAE_std': 0.0,
            'R2': row['R2'],     'R2_std': 0.0,
            'n_seeds': 1, 'base_seed': base_seed,
        })

    # Стохастические — агрегируем
    if not df_stoch.empty:
        agg = (df_stoch
               .groupby(['target', 'scenario', 'forecast_model', 'delays'])
               .agg(RMSE_mean=('RMSE', 'mean'), RMSE_std=('RMSE', 'std'),
                    MAE_mean=('MAE', 'mean'),   MAE_std=('MAE', 'std'),
                    R2_mean=('R2', 'mean'),     R2_std=('R2', 'std'))
               .reset_index())
        for _, row in agg.iterrows():
            records.append({
                'target': row['target'],
                'delays': row['delays'],
                'scenario': row['scenario'],
                'forecast_model': row['forecast_model'],
                'RMSE': row['RMSE_mean'], 'RMSE_std': row['RMSE_std'],
                'MAE': row['MAE_mean'],   'MAE_std': row['MAE_std'],
                'R2': row['R2_mean'],     'R2_std': row['R2_std'],
                'n_seeds': n_seeds, 'base_seed': base_seed,
            })

    df_agg = pd.DataFrame(records)
    results_list.extend(records)

    if verbose:
        for _, row in df_agg.iterrows():
            print(f"  {row['forecast_model']:8s}: "
                  f"RMSE = {row['RMSE']:.4f} ± {row['RMSE_std']:.4f}, "
                  f"MAE = {row['MAE']:.4f} ± {row['MAE_std']:.4f}, "
                  f"R2 = {row['R2']:.4f} ± {row['R2_std']:.4f}")

    df_all = pd.concat([df_single, df_stoch], ignore_index=True) if not df_stoch.empty else df_single
    return df_all, df_agg, results_list


# =====================================================================
# 4. ФУНКЦИИ ДЛЯ ЗАПУСКА СЦЕНАРИЕВ
# =====================================================================

def run_no_adaptation_scenario(ace_df, disc_df, split_date, target, future_lags,
                               scenario, build_models_fn,
                               delays='24h', n_seeds=3, base_seed=42,
                               results_list=None, verbose=True):
    """ Запуск одного сценария БЕЗ адаптации (ace-ace, disc-disc, ace-disc, disc-ace) """
    train, test = split_datasets_by_scenario(ace_df, disc_df, split_date, scenario)
    return evaluate_with_seeds(
        train, test, target, future_lags, build_models_fn,
        scenario_label=scenario, delays=delays,
        n_seeds=n_seeds, base_seed=base_seed,
        results_list=results_list, verbose=verbose,
    )


def run_combined_scenario(ace_df, disc_df, split_date, target, future_lags,
                          build_models_fn, delays='24h',
                          n_seeds=3, base_seed=42,
                          results_list=None, verbose=True):
    """
    Сценарий комбинированных данных: обучение на ACE+DSCOVR с флагом is_ace
    """
    train, test = make_combined_train(ace_df, disc_df, split_date, add_is_ace_flag=True)
    return evaluate_with_seeds(
        train, test, target, future_lags, build_models_fn,
        scenario_label='combined', delays=delays,
        n_seeds=n_seeds, base_seed=base_seed,
        results_list=results_list, verbose=verbose,
    )


def run_adaptation_D2A(ace_df, disc_test_adapted, split_date, target, future_lags,
                       build_models_fn, adaptation_label,
                       delays='24h', n_seeds=3, base_seed=42,
                       results_list=None, verbose=True):
    """
    Сценарий ДА DSCOVR->ACE: M_A обучается на ACE_train, тестируется на адаптированном DSCOVR_test
    """
    split_date = pd.Timestamp(split_date)
    ace_train = ace_df.loc[:split_date]
    return evaluate_with_seeds(
        ace_train, disc_test_adapted, target, future_lags, build_models_fn,
        scenario_label=adaptation_label, delays=delays,
        n_seeds=n_seeds, base_seed=base_seed,
        results_list=results_list, verbose=verbose,
    )


def run_adaptation_A2D(ace_old_adapted, disc_train, disc_test, target, future_lags,
                       build_models_fn, adaptation_label,
                       delays='24h', n_seeds=3, base_seed=42,
                       results_list=None, verbose=True):
    """
    Сценарий ДА ACE->DSCOVR: M_D обучается на (адаптированный старый ACE + DSCOVR_train),
    тестируется на DSCOVR_test
    """
    train = pd.concat([ace_old_adapted, disc_train]).sort_index()
    return evaluate_with_seeds(
        train, disc_test, target, future_lags, build_models_fn,
        scenario_label=adaptation_label, delays=delays,
        n_seeds=n_seeds, base_seed=base_seed,
        results_list=results_list, verbose=verbose,
    )
