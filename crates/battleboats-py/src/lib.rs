//! Python module `battleboats_core`.
//!
//! Importing this fails only when the extension was not built. The pure-Python
//! engine does not import this module.

use bb_engine::fog::{CommanderRng, FogCommander as CoreFog, FogConfig};
use bb_engine::{
    canonical_json, decode_json, format_python_json_float, state_from_canonical_json, state_hash,
    tables, Action, CoreError, Engine, ShipType, SpotKey, CASH_PER_MATERIAL, MERCHANT_CAPACITY,
    PORT_PRODUCTION, SCHEMA_VERSION, STARTING_CASH,
};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

fn core_err(err: CoreError) -> PyErr {
    if err.message.contains("outside") {
        PyIndexError::new_err(err.message)
    } else {
        PyValueError::new_err(err.message)
    }
}

fn py_dumps(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    let dumps = obj.py().import("json")?.getattr("dumps")?;
    dumps.call1((obj,))?.extract()
}

fn pos_pair<'py>(py: Python<'py>, pos: (i32, i32)) -> PyResult<Bound<'py, PyList>> {
    let pair = PyList::empty(py);
    pair.append(pos.0)?;
    pair.append(pos.1)?;
    Ok(pair)
}

fn action_list<'py>(py: Python<'py>, action: &Action) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    match *action {
        Action::Move { ship_id, x, y } => {
            list.append("m")?;
            list.append(ship_id)?;
            list.append(x)?;
            list.append(y)?;
        }
        Action::Attack {
            attacker_id,
            target_id,
        } => {
            list.append("a")?;
            list.append(attacker_id)?;
            list.append(target_id)?;
        }
        Action::BuildShip {
            port_x,
            port_y,
            spawn_x,
            spawn_y,
            ship_type,
        } => {
            list.append("b")?;
            list.append(port_x)?;
            list.append(port_y)?;
            list.append(spawn_x)?;
            list.append(spawn_y)?;
            list.append(ship_type.value())?;
        }
        Action::BuildPort {
            builder_ship_id,
            x,
            y,
        } => {
            list.append("p")?;
            list.append(builder_ship_id)?;
            list.append(x)?;
            list.append(y)?;
        }
        Action::CapturePort {
            landing_ship_id,
            x,
            y,
        } => {
            list.append("c")?;
            list.append(landing_ship_id)?;
            list.append(x)?;
            list.append(y)?;
        }
        Action::MerchantLoad { merchant_id, x, y } => {
            list.append("l")?;
            list.append(merchant_id)?;
            list.append(x)?;
            list.append(y)?;
        }
        Action::MerchantUnload { merchant_id, x, y } => {
            list.append("u")?;
            list.append(merchant_id)?;
            list.append(x)?;
            list.append(y)?;
        }
        Action::EndTurn => list.append("e")?,
    }
    Ok(list)
}

fn player_or_err(
    state: &bb_engine::GameState,
    player_id: i64,
) -> PyResult<&bb_engine::state::Player> {
    state
        .players
        .iter()
        .find(|player| player.id == player_id)
        .ok_or_else(|| PyValueError::new_err(format!("no player {player_id}")))
}

fn port_views<'py>(
    engine: &Engine,
    py: Python<'py>,
    player_id: i64,
    fresh_only: bool,
) -> PyResult<Bound<'py, PyList>> {
    let state = engine.require().map_err(core_err)?;
    let player = player_or_err(state, player_id)?;
    let mut rows: Vec<_> = player.port_sightings.values().collect();
    rows.sort_by_key(|s| s.position);
    let list = PyList::empty(py);
    for sighting in rows {
        if fresh_only && !sighting.fresh {
            continue;
        }
        let dict = PyDict::new(py);
        dict.set_item("position", pos_pair(py, sighting.position)?)?;
        dict.set_item("last_known_owner", sighting.last_known_owner)?;
        dict.set_item("is_home", sighting.is_home)?;
        dict.set_item("turn_seen", sighting.turn_seen)?;
        dict.set_item("fresh", sighting.fresh)?;
        list.append(dict)?;
    }
    Ok(list)
}

fn pos_list<'py>(py: Python<'py>, cells: &[(i32, i32)]) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for &(x, y) in cells {
        let pair = PyList::empty(py);
        pair.append(x)?;
        pair.append(y)?;
        list.append(pair)?;
    }
    Ok(list)
}

struct PyRng<'a> {
    inner: &'a Bound<'a, PyAny>,
}

impl CommanderRng for PyRng<'_> {
    fn random(&mut self) -> f64 {
        self.inner
            .call_method0("random")
            .unwrap()
            .extract()
            .unwrap()
    }
    fn randrange(&mut self, n: usize) -> usize {
        self.inner
            .call_method1("randrange", (n,))
            .unwrap()
            .extract()
            .unwrap()
    }
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        self.inner
            .call_method1("uniform", (lo, hi))
            .unwrap()
            .extract()
            .unwrap()
    }
}

#[pyclass(name = "FogCommander")]
struct FogCommander {
    inner: CoreFog,
}

#[pymethods]
impl FogCommander {
    #[new]
    #[pyo3(signature = (doctrine = None))]
    fn new(doctrine: Option<String>) -> Self {
        let mut config = FogConfig::default();
        config.doctrine = doctrine;
        Self {
            inner: CoreFog::new(config),
        }
    }

    fn doctrine_for(&self, pid: i64) -> Option<String> {
        self.inner.doctrine_name(pid).map(str::to_string)
    }

    fn act<'py>(
        &mut self,
        py: Python<'py>,
        game: &Game,
        pid: i64,
        rng: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        let state = game.inner.require().map_err(core_err)?;
        let mut py_rng = PyRng { inner: rng };
        let action = self.inner.act(state, pid, &mut py_rng);
        action_list(py, &action)
    }
}

#[pyclass]
struct Batch {
    games: Vec<Engine>,
}

#[pymethods]
impl Batch {
    #[new]
    fn new(map_paths: Vec<String>, kill_curve_k: f64) -> Self {
        Self {
            games: map_paths
                .into_iter()
                .map(|path| Engine::from_map_path(path, kill_curve_k))
                .collect(),
        }
    }

    fn reset(&mut self, seeds: Vec<i64>) -> PyResult<()> {
        for (game, seed) in self.games.iter_mut().zip(seeds) {
            game.reset(Some(seed)).map_err(core_err)?;
        }
        Ok(())
    }

    fn step(&mut self, index: usize, action: &Bound<'_, PyAny>) -> PyResult<()> {
        let text = py_dumps(action)?;
        let decoded = decode_json(&text).map_err(core_err)?;
        self.games
            .get_mut(index)
            .ok_or_else(|| PyValueError::new_err("batch index out of range"))?
            .step(&decoded)
            .map_err(core_err)
    }

    fn state_hash(&self, index: usize) -> PyResult<String> {
        let game = self
            .games
            .get(index)
            .ok_or_else(|| PyValueError::new_err("batch index out of range"))?;
        Ok(state_hash(game.require().map_err(core_err)?))
    }

    fn len(&self) -> usize {
        self.games.len()
    }
}

#[pyclass(name = "Game")]
struct Game {
    inner: Engine,
}

#[pymethods]
impl Game {
    #[new]
    #[pyo3(signature = (map_path, kill_curve_k = 2.0))]
    fn new(map_path: &str, kill_curve_k: f64) -> Self {
        Self {
            inner: Engine::from_map_path(map_path, kill_curve_k),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (text, kill_curve_k = 2.0))]
    fn from_map_json(text: &str, kill_curve_k: f64) -> Self {
        Self {
            inner: Engine::from_map_json(text, kill_curve_k),
        }
    }

    #[staticmethod]
    fn from_canonical_json(text: &str) -> PyResult<Self> {
        let state = state_from_canonical_json(text).map_err(core_err)?;
        let kill_curve_k = state.kill_curve_k;
        Ok(Self {
            inner: Engine::from_state(state, kill_curve_k),
        })
    }

    #[pyo3(signature = (seed = None))]
    fn reset(&mut self, seed: Option<i64>) -> PyResult<()> {
        self.inner.reset(seed).map_err(core_err)?;
        Ok(())
    }

    fn reset_from_scenario(&mut self, scenario: &Bound<'_, PyAny>) -> PyResult<()> {
        let text = py_dumps(scenario)?;
        self.inner
            .reset_from_scenario_json(&text)
            .map_err(core_err)?;
        Ok(())
    }

    fn step(&mut self, action: &Bound<'_, PyAny>) -> PyResult<()> {
        let text = py_dumps(action)?;
        let decoded = decode_json(&text).map_err(core_err)?;
        self.inner.step(&decoded).map_err(core_err)
    }

    fn enumerate_legal<'py>(
        &self,
        py: Python<'py>,
        player_id: i64,
    ) -> PyResult<Bound<'py, PyList>> {
        let actions = self.inner.enumerate_legal(player_id).map_err(core_err)?;
        let list = PyList::empty(py);
        for action in actions {
            list.append(action_list(py, &action)?)?;
        }
        Ok(list)
    }

    fn is_terminal(&self) -> PyResult<bool> {
        self.inner.is_terminal().map_err(core_err)
    }

    fn turn(&self) -> PyResult<i64> {
        Ok(self.inner.require().map_err(core_err)?.turn)
    }

    fn current_player(&self) -> PyResult<i64> {
        Ok(self.inner.require().map_err(core_err)?.current_player)
    }

    fn winner(&self) -> PyResult<Option<i64>> {
        Ok(self.inner.require().map_err(core_err)?.winner)
    }

    #[pyo3(name = "clone")]
    fn clone_game(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    fn set_combat_outcomes(&mut self, pairs: &Bound<'_, PyAny>) -> PyResult<()> {
        let text = py_dumps(pairs)?;
        let raw: Vec<Vec<serde_json::Value>> =
            serde_json::from_str(&text).map_err(|err| PyValueError::new_err(err.to_string()))?;
        let mut outcomes = Vec::with_capacity(raw.len());
        for item in raw {
            let step = item
                .first()
                .and_then(serde_json::Value::as_i64)
                .ok_or_else(|| PyValueError::new_err("combat outcome step is not an int"))?;
            let killed = item
                .get(1)
                .and_then(serde_json::Value::as_bool)
                .ok_or_else(|| PyValueError::new_err("combat outcome killed is not a bool"))?;
            outcomes.push((step, killed));
        }
        self.inner.set_combat_outcomes(outcomes);
        Ok(())
    }

    fn visible_enemy_ships<'py>(
        &self,
        py: Python<'py>,
        player_id: i64,
    ) -> PyResult<Bound<'py, PyList>> {
        let state = self.inner.require().map_err(core_err)?;
        let player = player_or_err(state, player_id)?;
        let mut rows: Vec<_> = player
            .sightings
            .values()
            .filter(|s| s.fresh && state.ships.contains_key(&s.ship_id))
            .collect();
        rows.sort_by_key(|s| s.ship_id);
        let list = PyList::empty(py);
        for sighting in rows {
            let dict = PyDict::new(py);
            dict.set_item("id", sighting.ship_id)?;
            dict.set_item("type", sighting.ship_type.value())?;
            dict.set_item("position", pos_pair(py, sighting.position)?)?;
            list.append(dict)?;
        }
        Ok(list)
    }

    fn known_enemy_ships<'py>(
        &self,
        py: Python<'py>,
        player_id: i64,
    ) -> PyResult<Bound<'py, PyList>> {
        let state = self.inner.require().map_err(core_err)?;
        let player = player_or_err(state, player_id)?;
        let mut rows: Vec<_> = player.sightings.values().collect();
        rows.sort_by_key(|s| s.ship_id);
        let list = PyList::empty(py);
        for sighting in rows {
            let dict = PyDict::new(py);
            dict.set_item("ship_id", sighting.ship_id)?;
            dict.set_item("type", sighting.ship_type.value())?;
            dict.set_item("position", pos_pair(py, sighting.position)?)?;
            dict.set_item("turn_seen", sighting.turn_seen)?;
            dict.set_item("fresh", sighting.fresh)?;
            list.append(dict)?;
        }
        Ok(list)
    }

    fn visible_enemy_ports<'py>(
        &self,
        py: Python<'py>,
        player_id: i64,
    ) -> PyResult<Bound<'py, PyList>> {
        port_views(&self.inner, py, player_id, true)
    }

    fn observation<'py>(
        &self,
        py: Python<'py>,
        player_id: i64,
    ) -> PyResult<(Vec<f32>, Vec<f32>, usize)> {
        let state = self.inner.require().map_err(core_err)?;
        let obs = state.observation(player_id);
        let globals = obs.globals.to_vec();
        Ok((obs.tokens, globals, obs.n_tokens))
    }

    fn known_enemy_ports<'py>(
        &self,
        py: Python<'py>,
        player_id: i64,
    ) -> PyResult<Bound<'py, PyList>> {
        port_views(&self.inner, py, player_id, false)
    }

    fn canonical_json(&self) -> PyResult<String> {
        let state = self.inner.require().map_err(core_err)?;
        Ok(canonical_json(state))
    }

    fn state_hash(&self) -> PyResult<String> {
        let state = self.inner.require().map_err(core_err)?;
        Ok(state_hash(state))
    }

    #[getter]
    fn width(&self) -> PyResult<i32> {
        Ok(self.inner.require().map_err(core_err)?.map.width)
    }

    #[getter]
    fn height(&self) -> PyResult<i32> {
        Ok(self.inner.require().map_err(core_err)?.map.height)
    }

    #[getter]
    fn rng_seed(&self) -> Option<i64> {
        self.inner.rng_seed
    }

    #[getter]
    fn map_seed(&self) -> PyResult<Option<i64>> {
        Ok(self.inner.require().map_err(core_err)?.map.seed)
    }

    fn terrain_land<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let state = self.inner.require().map_err(core_err)?;
        pos_list(py, &state.map.terrain_land())
    }

    fn home_ports<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let state = self.inner.require().map_err(core_err)?;
        let homes: Vec<(i32, i32)> = state.players.iter().map(|p| p.home_port).collect();
        pos_list(py, &homes)
    }

    fn in_bounds(&self, x: i32, y: i32) -> PyResult<bool> {
        Ok(self.inner.require().map_err(core_err)?.map.in_bounds(x, y))
    }

    fn is_water(&self, x: i32, y: i32) -> PyResult<bool> {
        self.inner
            .require()
            .map_err(core_err)?
            .map
            .is_water(x, y)
            .map_err(core_err)
    }

    fn is_land(&self, x: i32, y: i32) -> PyResult<bool> {
        self.inner
            .require()
            .map_err(core_err)?
            .map
            .is_land(x, y)
            .map_err(core_err)
    }

    fn is_port(&self, x: i32, y: i32) -> PyResult<bool> {
        self.inner
            .require()
            .map_err(core_err)?
            .map
            .is_port(x, y)
            .map_err(core_err)
    }

    fn port_owner(&self, x: i32, y: i32) -> PyResult<i64> {
        let owner = self
            .inner
            .require()
            .map_err(core_err)?
            .map
            .port_owner_at(x, y)
            .map_err(core_err)?;
        Ok(i64::from(owner))
    }

    fn manhattan(&self, x1: i32, y1: i32, x2: i32, y2: i32) -> i32 {
        bb_engine::Map::manhattan((x1, y1), (x2, y2))
    }

    fn tiles_within<'py>(
        &self,
        py: Python<'py>,
        x: i32,
        y: i32,
        r: i32,
    ) -> PyResult<Bound<'py, PyList>> {
        let state = self.inner.require().map_err(core_err)?;
        pos_list(py, &state.map.tiles_within(x, y, r))
    }

    /// Land tiles next to water, ports included. Matches `_coastline_scan`.
    fn coastline_scan<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let state = self.inner.require().map_err(core_err)?;
        pos_list(py, &state.map.coastline_owned())
    }

    /// `player_id=None` matches `observation._coastline_tiles(engine, None)`.
    #[pyo3(signature = (player_id = None))]
    fn coastline<'py>(
        &self,
        py: Python<'py>,
        player_id: Option<i64>,
    ) -> PyResult<Bound<'py, PyList>> {
        let state = self.inner.require().map_err(core_err)?;
        let cells = state.coastline_tiles(player_id).map_err(core_err)?;
        pos_list(py, &cells)
    }
}

#[pyfunction]
fn schema_version() -> i64 {
    SCHEMA_VERSION
}

#[pyfunction]
fn starting_cash() -> i64 {
    STARTING_CASH
}

#[pyfunction]
fn port_production() -> i64 {
    PORT_PRODUCTION
}

#[pyfunction]
fn cash_per_material() -> i64 {
    CASH_PER_MATERIAL
}

#[pyfunction]
fn merchant_capacity() -> i64 {
    MERCHANT_CAPACITY
}

#[pyfunction]
fn ship_types() -> Vec<&'static str> {
    ShipType::ALL.iter().map(|t| t.value()).collect()
}

#[pyfunction]
fn base_stats<'py>(py: Python<'py>, ship_type: &str) -> PyResult<Bound<'py, PyDict>> {
    let kind = ShipType::from_value(ship_type).map_err(core_err)?;
    let stats = tables().stats(kind);
    let dict = PyDict::new(py);
    dict.set_item("speed", stats.speed)?;
    dict.set_item("cost", stats.cost)?;
    dict.set_item("strength", stats.strength)?;
    dict.set_item("attack_range", stats.attack_range)?;
    Ok(dict)
}

#[pyfunction]
fn attack_modifier(attacker_type: &str, defender_type: &str) -> PyResult<f64> {
    let attacker = ShipType::from_value(attacker_type).map_err(core_err)?;
    let defender = ShipType::from_value(defender_type).map_err(core_err)?;
    Ok(tables().attack_modifier(attacker, defender))
}

#[pyfunction]
fn spot_distance(spotter_key: &str, target_key: &str) -> PyResult<f64> {
    let spotter = SpotKey::parse(spotter_key).map_err(core_err)?;
    let target = SpotKey::parse(target_key).map_err(core_err)?;
    Ok(tables().spot_distance(spotter, target))
}

#[pyfunction]
fn format_canonical_float(value: f64) -> String {
    format_python_json_float(value)
}

#[pyfunction]
fn canonical_json_from_json(text: &str) -> PyResult<String> {
    let state = state_from_canonical_json(text).map_err(core_err)?;
    Ok(canonical_json(&state))
}

#[pyfunction]
fn state_hash_from_json(text: &str) -> PyResult<String> {
    let state = state_from_canonical_json(text).map_err(core_err)?;
    Ok(state_hash(&state))
}

#[pyfunction]
fn roundtrip_action(rec: &Bound<'_, PyAny>) -> PyResult<String> {
    let text = py_dumps(rec)?;
    let action = decode_json(&text).map_err(core_err)?;
    Ok(action.encode_json())
}

#[pymodule]
fn battleboats_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Fail at import if the balance CSVs baked into the binary do not parse.
    let _ = tables();
    m.add_class::<Game>()?;
    m.add_class::<FogCommander>()?;
    m.add_class::<Batch>()?;
    m.add_function(wrap_pyfunction!(schema_version, m)?)?;
    m.add_function(wrap_pyfunction!(starting_cash, m)?)?;
    m.add_function(wrap_pyfunction!(port_production, m)?)?;
    m.add_function(wrap_pyfunction!(cash_per_material, m)?)?;
    m.add_function(wrap_pyfunction!(merchant_capacity, m)?)?;
    m.add_function(wrap_pyfunction!(ship_types, m)?)?;
    m.add_function(wrap_pyfunction!(base_stats, m)?)?;
    m.add_function(wrap_pyfunction!(attack_modifier, m)?)?;
    m.add_function(wrap_pyfunction!(spot_distance, m)?)?;
    m.add_function(wrap_pyfunction!(format_canonical_float, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_json_from_json, m)?)?;
    m.add_function(wrap_pyfunction!(state_hash_from_json, m)?)?;
    m.add_function(wrap_pyfunction!(roundtrip_action, m)?)?;
    Ok(())
}
