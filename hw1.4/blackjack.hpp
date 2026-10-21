// ============================================================================
// blackjack.hpp —— 21 点（Blackjack）游戏环境
//
// 对应 Sutton《强化学习》Example 5.1 / 5.3 的环境部分，提供 gym 风格接口：
//     reset() / reset(,,)  开始一局（随机开局 / 指定初始状态开局）
//     step(action)         执行一个动作，返回 StepResult(新状态, 奖励, 是否结束)
// 规则要点：
//   - 玩家（智能体）先行动，玩家停牌后轮到庄家行动（策略由调用方驱动）
//   - A 可作 1 或 11（usable ace）；J/Q/K 均按 10 计
//   - 奖励只在终局给出：玩家胜 +1，负 -1，平局 0
// ============================================================================
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <utility>
#include <algorithm>
#include <iostream>

using namespace std;

// 21 点环境类：封装一局游戏的全部状态与规则推进
class Blackjack
{
public:
    bool verbose; // 是否打印每步详细信息（演示用；训练时应置 false）
    // ---- 身份标识：当前轮到谁行动（存于 State.turn） ----
    // identities
    // 翻译：DEALER 庄家, PLAYER 玩家;
    static const bool DEALER = false, PLAYER = true;
    // ---- 动作标识：每回合的两个可选动作（作为 step 的入参） ----
    // actions
    static const int HIT = 0, STICK = 1; // HIT 要牌；STICK 停牌
    // 牌面打印字符表：下标即牌值（T 表示 10；J/Q/K 发牌时已折算为 10）
    static constexpr const char *CARD_NAME = " A23456789TJQK";
    // ---- 游戏状态（完整 MDP 状态） ----
    // 注：对智能体（玩家）而言，决策能依据的观测只有三元组
    //     (dealer_shown, player_ace, player_sum)，庄家暗牌不可见
    struct State
    {
        // DEALER or PLAYER's turn：当前是庄家还是玩家的回合
        bool turn;
        // if has usable ace then true：玩家/庄家手中是否有"可用 A"（按 11 计且未爆）
        bool player_ace, dealer_ace;
        // 0-21 or >21 (goes bust)：玩家/庄家当前点数，>21 即爆点
        int player_sum, dealer_sum;
        // dealer's shown card: [A,10]：庄家明牌（玩家唯一可见的庄家信息）
        int dealer_shown;
    };
    // ---- 一步交互的返回结果（gym 风格） ----
    struct StepResult
    {
        // state after the step：执行动作后的新状态
        State state;
        // reward of the step. winner gets 1, loser gets -1, tie or non-terminal step both get 0.
        // 本步奖励：终局时胜者 +1、负者 -1；平局或非终局步均为 0
        double player_reward, dealer_reward;
        // whether the game terminates after the step：本步之后一局是否结束
        bool done;
        StepResult(const State &state, double player_reward, double dealer_reward, bool done)
        {
            this->state = state;
            this->player_reward = player_reward;
            this->dealer_reward = dealer_reward;
            this->done = done;
        }
        StepResult() {}
    };
    // ---- 随机开局：给庄家、玩家各随机发两张牌 ----
    // restart the game with cards randomly allocated to player and dealer
    void reset(void);

    // ---- 指定初始状态开局：固定玩家观测三元组，庄家暗牌随机补抽 ----
    // 供蒙特卡洛 Exploring Starts 使用：枚举所有 (dealer_shown, player_ace, player_sum)
    // restart the game with given initial state (observation of the player)
    void reset(int dealer_shown, bool player_ace, int player_sum);

    // 读取当前状态（返回副本）
    State state(void) const
    {
        return _state;
    }
    // if sum > 21 then goes bust：当前行动方（turn 指向的一方）是否爆点
    bool goes_bust(void) const
    {
        return _state.turn == PLAYER ? _state.player_sum > 21 : _state.dealer_sum > 21;
    }
    // ---- 推进游戏一步：动作作用于当前 turn 指向的一方 ----
    // action: HIT(0) 要牌 or STICK(1) 停牌
    StepResult step(int action);

    Blackjack(bool verbose = false)
    {
        this->verbose = verbose;
        srand(time(nullptr)); // 以当前时间为随机种子（发牌用）
    }

private:
    // natural means player wins or ties with only 2 cards which sums to 21.
    // natural 标记：本局是否仍处于"开局两张牌"阶段；玩家开局即 21 点会直接终局结算
    bool natural;
    // current state of the game：当前游戏状态
    State _state;
    // 随机抽一张牌：A~K 等概率抽取，但 J/Q/K 都折算成 10（故 10 的概率为 4/13）
    // sample a card randomly from A to K. J,Q,Ks are regarded as 10.
    int sample_card()
    {
        return min(rand() % 13 + 1, 10); //[A,2,3,4,5,6,7,8,9,10,J,Q,K]
    }
    // 把抽到的牌发给当前 turn 指向的一方，并处理 A 的 1/11 取值
    // sample a card and give it to player/dealer who HIT.
    void deal_card(int card);
};

// =========================== reset()：随机开局 ===========================
void Blackjack::reset()
{
    natural = true; // 进入"开局两张牌"阶段
    _state.player_sum = 0;
    _state.dealer_sum = 0;
    _state.player_ace = false;
    _state.dealer_ace = false;
    // 先给庄家发两张：第一张为明牌（玩家可见），第二张为暗牌
    _state.turn = DEALER;
    int card1 = sample_card(), card2 = sample_card();
    _state.dealer_shown = card1; // 记录庄家明牌
    deal_card(card1);
    deal_card(card2);
    if (verbose)
    {
        cout << "Game restarted." << endl;
        cout << "Dealer allocated\n\tPublic: " << CARD_NAME[card1]
             << "\n\tPrivate: " << CARD_NAME[card2] << endl;
        cout << "Dealer:\n\tSum: " << _state.dealer_sum
             << "\n\tUsable ace: " << _state.dealer_ace << endl;
    }
    // 再给玩家发两张，此后进入玩家回合
    _state.turn = PLAYER;
    card1 = sample_card();
    card2 = sample_card();
    deal_card(card1);
    deal_card(card2);
    if (verbose)
    {
        cout << "Player allocated\n\t" << CARD_NAME[card1] << "\n\t" << CARD_NAME[card2] << endl;
        cout << "Player:\n\tSum:" << _state.player_sum
             << "\n\tUsable ace: " << _state.player_ace << endl;
    }
}

// ================ reset(,,)：指定初始状态开局（Exploring Starts 用） ================
// 固定玩家的观测状态 (dealer_shown, player_ace, player_sum)，
// 庄家暗牌随机补抽一张，构成合法开局；之后直接进入玩家回合。
void Blackjack::reset(int dealer_shown, bool player_ace, int player_sum)
{
    natural = true;
    _state.dealer_shown = dealer_shown;
    _state.player_ace = player_ace;
    _state.player_sum = player_sum;
    // 补全庄家手牌：明牌已知，暗牌随机抽一张
    _state.turn = DEALER;
    int dealer_another_card = sample_card();
    _state.dealer_sum = dealer_shown + dealer_another_card;
    // 两张中若有 A 且按 11 计不爆，则记为可用 A（点数额外 +10）
    if (dealer_another_card == 1 or dealer_shown == 1)
    {
        _state.dealer_sum += 10;
        _state.dealer_ace = true;
    }
    else
    {
        _state.dealer_ace = false;
    }
    // 进入玩家回合：智能体从这个指定状态开始决策
    _state.turn = PLAYER;
    if (verbose)
    {
        cout << "<State fixed reset:> " << endl;
        cout << "Player" << endl;
        cout << "\n\tSum: " << _state.player_sum << "\n\tAce: " << _state.player_ace << endl;
        cout << "Dealer: " << endl;
        cout << "\n\tSum: " << _state.dealer_sum << "\n\tAce: "
             << _state.dealer_ace << "\n\tShown: " << _state.dealer_shown << endl;
    }
}

// ========================= step(action)：推进游戏一步 =========================
Blackjack::StepResult Blackjack::step(int action)
{
    // ---- 开局结算：natural（天牌，开局两张即 21 点） ----
    // 玩家开局即 21：庄家也是 21 则平局，否则玩家直接获胜，游戏立即结束
    if (natural)
    {
        if (_state.player_sum == 21)
        {
            if (_state.dealer_sum == 21)
            {
                if (verbose)
                {
                    cout << "Natural tie." << endl;
                }
                return StepResult(_state, 0.0, 0.0, true); // 天牌平局
            }
            else
            {
                if (verbose)
                {
                    cout << "Player natural." << endl;
                }
                return StepResult(_state, 1.0, -1.0, true); // 玩家天牌获胜
            }
        }
        else
        {
            natural = false; // 玩家开局未到 21，之后按正常流程进行
        }
    }
    const char *identity = (_state.turn == PLAYER ? "Player" : "Dealer");
    if (action == HIT)
    {
        // ---- 要牌：随机抽一张发给当前行动方 ----
        int card = sample_card();
        deal_card(card);
        int current_sum = _state.turn == PLAYER ? _state.player_sum : _state.dealer_sum;
        if (verbose)
        {
            cout << identity << " Hit\n\t" << CARD_NAME[card] << endl;
            cout << "After hit: " << identity << "\n\tSum: " << current_sum
                 << "\n\tUsable ace: " << (_state.turn == PLAYER ? _state.player_ace : _state.dealer_ace) << endl;
        }
        if (goes_bust())
        {
            // 当前行动方爆点 → 立即终局，对方获胜
            double player_reward = _state.turn == PLAYER ? -1.0 : 1.0;
            if (verbose)
            {
                cout << identity << " goes bust." << endl;
            }
            return StepResult(_state, player_reward, -player_reward, true);
        }
        else
        {
            // 未爆点 → 游戏继续（回合不变，同一方还可继续行动）
            return StepResult(_state, 0.0, 0.0, false);
        }
    }
    else
    {
        // ---- 停牌 ----
        switch (_state.turn)
        {
        case PLAYER:
            // 玩家停牌 → 不移交胜负判定，只把回合交给庄家
            if (verbose)
            {
                cout << "Player stick at " << _state.player_sum << ".\nDealer's turn." << endl;
            }
            _state.turn = DEALER;
            return StepResult(_state, 0.0, 0.0, false);
        case DEALER:
            // 庄家停牌 → 游戏结束，比较双方点数定胜负
            if (verbose)
            {
                cout << "Dealer stick at " << _state.dealer_sum << ".\nGame done." << endl;
            }
            if (_state.player_sum == _state.dealer_sum)
            {
                // 点数相同：平局
                if (verbose)
                {
                    cout << "Game tie, player_sum = dealer_sum = " << _state.player_sum << endl;
                }
                return StepResult(_state, 0.0, 0.0, true);
            }
            else
            {
                // 点数大者胜
                double player_reward = _state.player_sum > _state.dealer_sum ? 1.0 : -1.0;
                if (verbose)
                {
                    cout << (player_reward > 0 ? "Player" : "Dealer") << " win.\n\tPlayer: " << _state.player_sum << "\n\tDealer: "
                         << _state.dealer_sum << endl;
                }
                return StepResult(_state, player_reward, -player_reward, true);
            }
        }
    }
}

// ================== deal_card：给当前行动方发牌，并处理 A 的取值 ==================
void Blackjack::deal_card(int card)
{
    // 用引用直接修改当前行动方（turn 指向的一方）的点数与可用 A 标记
    bool &useable_ace = _state.turn == PLAYER ? _state.player_ace : _state.dealer_ace;
    int &current_sum = _state.turn == PLAYER ? _state.player_sum : _state.dealer_sum;
    if (card == 1)
    {
        // 抽到 A：若按 11 计不爆则算 11（并标记 usable ace），否则只能算 1
        if (current_sum + 11 <= 21)
        {
            useable_ace = true;
            current_sum += 11;
        }
        else
        {
            current_sum += 1;
        }
    }
    else
    {
        // 非 A：直接加点
        current_sum += card;
        // 若因此爆点且手中有可用 A，把一个 A 从 11 退化为 1（总点数 -10）
        if (goes_bust() and useable_ace)
        {
            useable_ace = false;
            current_sum -= 11;
            current_sum += 1;
        }
    }
}