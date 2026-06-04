"""Self-improving growth layer for the agent.

エージェントの自己成長層.

接続層(main.py / starter.py / Client)には一切触れず, Agent クラスの内側からのみ
利用される. 発話・行動プロンプトへの「確定情報への接地」「自己スタンスのアンカリング」
「批判的読解の規範」の注入と, 役職別戦略skill, ゲーム後レビューを担う.
"""
