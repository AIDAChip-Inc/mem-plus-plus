"""Native off-policy solver: pooled training, corpus loading, fail-closed answers."""
import copy
import hashlib
from src.agent.mempp import MemppAgent, MemppAgentConfig, canonical
from src.solver.base import BaseSolver


class MemppSolver(BaseSolver):
    AGENT_CLASS = MemppAgent

    def __init__(self, config, memory_cache_dir):
        super().__init__(config, memory_cache_dir)
        self.method_name = 'Mem++'

    def create_or_load_memory(self, dialogs):
        # BaseSolver otherwise drops dataset identity when passing test_idx.
        scoped = [{**row, 'test_idx': canonical([row['dataset'], row['test_idx']])} for row in dialogs]
        return self._create_or_load_memory(scoped, can_thread=False)

    def memory_locomo_conversation(self, conversation, session_cnt):
        # Whole-conversation hash keeps distinct corpora's original turn IDs
        # apart; all their sessions still share the same task world.
        corpus_id = hashlib.sha256(canonical(conversation).encode()).hexdigest()
        for session in range(1, session_cnt + 1):
            date = conversation[f'session_{session}_date_time']
            for index, turn in enumerate(conversation[f'session_{session}']):
                key = canonical([corpus_id, session, turn['dia_id'], index])
                content = f"{date}\nTurn {turn['dia_id']}\nSpeaker {turn['speaker']} says: {turn['text']}"
                self.agent.add_memory(content, 'corpus:' + hashlib.sha256(key.encode()).hexdigest())
        self.agent.save_memories()

    memory_dialsim_conversation = memory_locomo_conversation

    def predict_single_data(self, dataset, data):
        # Unlike the general solver fallback, infrastructure/API budget failures
        # must stop the run rather than become fake model-response strings.
        messages = copy.deepcopy(dataset.get_initial_chat_messages(data['test_idx']))
        self.agent.clear_last_memory_trace()
        response = self.agent.generate_response(messages, lang=data['lang'], retrieve_k=self.agent.config.retrieve_k)
        if not isinstance(response, str):
            raise ValueError('Model response must be text')
        return self._attach_memory_trace({'test_idx': data['test_idx'], 'messages': messages, 'response': response})
