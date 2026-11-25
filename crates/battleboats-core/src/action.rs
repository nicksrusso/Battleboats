//! Compact action encoding from `docs/golden_tests.md` §4 / `scripts/harvest_value.py`.

use crate::error::{CoreError, CoreResult};
use crate::ship::ShipType;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    Move {
        ship_id: i64,
        x: i32,
        y: i32,
    },
    Attack {
        attacker_id: i64,
        target_id: i64,
    },
    BuildShip {
        port_x: i32,
        port_y: i32,
        spawn_x: i32,
        spawn_y: i32,
        ship_type: ShipType,
    },
    BuildPort {
        builder_ship_id: i64,
        x: i32,
        y: i32,
    },
    CapturePort {
        landing_ship_id: i64,
        x: i32,
        y: i32,
    },
    MerchantLoad {
        merchant_id: i64,
        x: i32,
        y: i32,
    },
    MerchantUnload {
        merchant_id: i64,
        x: i32,
        y: i32,
    },
    EndTurn,
}

impl Action {
    pub fn encode_json(&self) -> String {
        match self {
            Action::Move { ship_id, x, y } => format!(r#"["m",{ship_id},{x},{y}]"#),
            Action::Attack {
                attacker_id,
                target_id,
            } => format!(r#"["a",{attacker_id},{target_id}]"#),
            Action::BuildShip {
                port_x,
                port_y,
                spawn_x,
                spawn_y,
                ship_type,
            } => format!(
                r#"["b",{},{},{},{},"{}"]"#,
                port_x,
                port_y,
                spawn_x,
                spawn_y,
                ship_type.value()
            ),
            Action::BuildPort {
                builder_ship_id,
                x,
                y,
            } => format!(r#"["p",{builder_ship_id},{x},{y}]"#),
            Action::CapturePort {
                landing_ship_id,
                x,
                y,
            } => format!(r#"["c",{landing_ship_id},{x},{y}]"#),
            Action::MerchantLoad { merchant_id, x, y } => {
                format!(r#"["l",{merchant_id},{x},{y}]"#)
            }
            Action::MerchantUnload { merchant_id, x, y } => {
                format!(r#"["u",{merchant_id},{x},{y}]"#)
            }
            Action::EndTurn => r#"["e"]"#.to_string(),
        }
    }
}

pub fn decode_json(text: &str) -> CoreResult<Action> {
    let value: serde_json::Value =
        serde_json::from_str(text).map_err(|e| CoreError::new(format!("action json: {e}")))?;
    let arr = value
        .as_array()
        .ok_or_else(|| CoreError::new("encoded action is not a list"))?;
    if arr.is_empty() {
        return Err(CoreError::new("encoded action is empty"));
    }
    let tag = arr[0]
        .as_str()
        .ok_or_else(|| CoreError::new("action tag is not a string"))?;
    match tag {
        "m" => Ok(Action::Move {
            ship_id: req_i64(arr, 1)?,
            x: req_i32(arr, 2)?,
            y: req_i32(arr, 3)?,
        }),
        "a" => Ok(Action::Attack {
            attacker_id: req_i64(arr, 1)?,
            target_id: req_i64(arr, 2)?,
        }),
        "b" => Ok(Action::BuildShip {
            port_x: req_i32(arr, 1)?,
            port_y: req_i32(arr, 2)?,
            spawn_x: req_i32(arr, 3)?,
            spawn_y: req_i32(arr, 4)?,
            ship_type: ShipType::from_value(req_str(arr, 5)?)?,
        }),
        "p" => Ok(Action::BuildPort {
            builder_ship_id: req_i64(arr, 1)?,
            x: req_i32(arr, 2)?,
            y: req_i32(arr, 3)?,
        }),
        "c" => Ok(Action::CapturePort {
            landing_ship_id: req_i64(arr, 1)?,
            x: req_i32(arr, 2)?,
            y: req_i32(arr, 3)?,
        }),
        "l" => Ok(Action::MerchantLoad {
            merchant_id: req_i64(arr, 1)?,
            x: req_i32(arr, 2)?,
            y: req_i32(arr, 3)?,
        }),
        "u" => Ok(Action::MerchantUnload {
            merchant_id: req_i64(arr, 1)?,
            x: req_i32(arr, 2)?,
            y: req_i32(arr, 3)?,
        }),
        "e" => Ok(Action::EndTurn),
        other => Err(CoreError::new(format!(
            "Unknown encoded action tag: {other:?}"
        ))),
    }
}

fn req_i64(arr: &[serde_json::Value], idx: usize) -> CoreResult<i64> {
    arr.get(idx)
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| CoreError::new(format!("action field {idx} is not an int")))
}

fn req_i32(arr: &[serde_json::Value], idx: usize) -> CoreResult<i32> {
    let n = req_i64(arr, idx)?;
    i32::try_from(n).map_err(|_| CoreError::new(format!("action field {idx} does not fit in i32")))
}

fn req_str(arr: &[serde_json::Value], idx: usize) -> CoreResult<&str> {
    arr.get(idx)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| CoreError::new(format!("action field {idx} is not a string")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tag_round_trips() {
        let actions = [
            Action::Move {
                ship_id: 3,
                x: 1,
                y: 4,
            },
            Action::Attack {
                attacker_id: 1,
                target_id: 9,
            },
            Action::BuildShip {
                port_x: 0,
                port_y: 0,
                spawn_x: 1,
                spawn_y: 0,
                ship_type: ShipType::Destroyer,
            },
            Action::BuildPort {
                builder_ship_id: 4,
                x: 2,
                y: 2,
            },
            Action::CapturePort {
                landing_ship_id: 5,
                x: 8,
                y: 1,
            },
            Action::MerchantLoad {
                merchant_id: 6,
                x: 3,
                y: 3,
            },
            Action::MerchantUnload {
                merchant_id: 6,
                x: 0,
                y: 0,
            },
            Action::EndTurn,
        ];
        let expected = [
            r#"["m",3,1,4]"#,
            r#"["a",1,9]"#,
            r#"["b",0,0,1,0,"Destroyer"]"#,
            r#"["p",4,2,2]"#,
            r#"["c",5,8,1]"#,
            r#"["l",6,3,3]"#,
            r#"["u",6,0,0]"#,
            r#"["e"]"#,
        ];
        for (action, json) in actions.iter().zip(expected) {
            let encoded = action.encode_json();
            assert_eq!(encoded, json);
            assert_eq!(&decode_json(&encoded).unwrap(), action);
        }
    }

    #[test]
    fn unknown_tag_is_rejected() {
        let err = decode_json(r#"["z"]"#).unwrap_err();
        assert!(err.message.contains("Unknown encoded action tag"));
    }
}
