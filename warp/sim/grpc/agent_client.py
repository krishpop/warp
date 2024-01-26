import grpc
import agent_pb2
import agent_pb2_grpc

import os


def run_client():
    channel = grpc.insecure_channel("localhost:50051")
    stub = agent_pb2_grpc.AgentStub(channel)
    filename = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets/nv_humanoid.xml"))
    model = agent_pb2.Model(xml=filename)
    response = stub.Init(agent_pb2.InitRequest(model=model, dt=1 / 100))
    print("Agent server response: ", response)


if __name__ == "__main__":
    run_client()
