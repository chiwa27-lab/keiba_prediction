from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Kaisai:
    kaisai_id: str
    kaisai_date: str
    place_code: str
    place_name: str


@dataclass
class RaceInfo:
    race_id: str
    race_date: str
    place_code: str
    place_name: str
    race_number: int
    race_name: str
    grade: str | None
    post_time: str | None
    surface: str | None
    distance: int | None
    direction: str | None
    weather: str | None
    track_condition: str | None
    kai: int | None
    nichi: int | None
    race_class: str | None
    num_horses: int | None
    prize_money: str | None
    corner_passing: str | None
    lap_times: str | None


@dataclass
class ResultEntry:
    race_id: str
    finish_position: int | None
    finish_status: str
    bracket_number: int | None
    horse_number: int | None
    horse_id: str | None
    horse_name: str
    sex: str | None
    age: int | None
    weight_carried: float | None
    jockey_id: str | None
    jockey_name: str | None
    finish_time: str | None
    finish_time_sec: float | None
    margin: str | None
    popularity: int | None
    win_odds: float | None
    last_3f: float | None
    trainer_affiliation: str | None
    trainer_id: str | None
    trainer_name: str | None
    horse_weight: int | None
    horse_weight_diff: int | None


@dataclass
class Payout:
    race_id: str
    bet_type: str
    combination: str
    payout_yen: int | None
    popularity: int | None


@dataclass
class Pedigree:
    """父・母・母の父。血統表を取得できなかった馬は horse_id 以外が None."""

    horse_id: str
    sire_id: str | None = None
    sire_name: str | None = None
    sire_line: str | None = None
    sire_sire_id: str | None = None
    sire_sire_name: str | None = None
    dam_id: str | None = None
    dam_name: str | None = None
    damsire_id: str | None = None
    damsire_name: str | None = None


@dataclass
class RaceResult:
    race: RaceInfo
    entries: list[ResultEntry] = field(default_factory=list)
    payouts: list[Payout] = field(default_factory=list)
