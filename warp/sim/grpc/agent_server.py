# python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. agent.proto

import grpc
from concurrent import futures
import time
import agent_pb2
import agent_pb2_grpc


class AgentServicer(agent_pb2_grpc.AgentServicer):
    def Init(self, request, context):
        """Initialize MJPC Agent."""
        print("request=", request)
        return agent_pb2.InitResponse(message="loaded successfully")

    def GetState(self, request, context):
        """Get the simulation state."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def SetState(self, request, context):
        """Set state of the MJPC Agent."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def Step(self, request, context):
        """Get the current action from the Agent.
        rpc GetAction(GetActionRequest) returns (GetActionResponse);
        Compute one plan step.
        rpc PlannerStep(PlannerStepRequest) returns (PlannerStepResponse);
        Step physics once, using actions from the planner.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def Reset(self, request, context):
        """Reset the Agent.
        // Set a task parameters.
        rpc SetTaskParameters(SetTaskParametersRequest)
        returns (SetTaskParametersResponse);
        // Get a task parameters.
        rpc GetTaskParameters(GetTaskParametersRequest)
        returns (GetTaskParametersResponse);
        Set cost weights.
        rpc SetCostWeights(SetCostWeightsRequest) returns (SetCostWeightsResponse);
        // Get cost term values.
        rpc GetCostValuesAndWeights(GetCostValuesAndWeightsRequest)
        returns (GetCostValuesAndWeightsResponse);
        // Set mode.
        rpc SetMode(SetModeRequest) returns (SetModeResponse);
        // Get mode.
        rpc GetMode(GetModeRequest) returns (GetModeResponse);
        // Get all modes.
        rpc GetAllModes(GetAllModesRequest) returns (GetAllModesResponse);
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    agent_pb2_grpc.add_AgentServicer_to_server(AgentServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Server started on port 50051...")
    try:
        while True:
            time.sleep(86400)  # One day in seconds
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()
