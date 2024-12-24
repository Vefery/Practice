using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class MazeAgent : Agent
{
    public float totalAngle = 360;
    public float numberRays = 32;
    private float delta;
    private RaycastHit hit;
    private CapsuleCollider capsuleCollider;
    private Maze maze;
    [System.NonSerialized]
    public int steps = 0;
    public Room currentRoom;

    protected override void Awake()
    {
        delta = totalAngle / numberRays;
        maze = FindFirstObjectByType<Maze>();
        capsuleCollider = GetComponent<CapsuleCollider>();
    }

    private void Update()
    {
        Vector3 pos = transform.position;
        const float magnitude = 5;

        for (int i = 0; i < numberRays; i++)
        {
            Vector3 dir = Quaternion.Euler(0, i * delta, 0) * transform.right;
            Debug.DrawLine(pos, pos + dir * magnitude);
        }
    }

    public override void OnEpisodeBegin()
    {
        Vector3 startPos = maze.startPose;

        maze.ResetMaze();
        steps = 0;
        currentRoom = maze.rooms[0];

        transform.position = startPos + new Vector3(0, capsuleCollider.height / 2, 0);
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        Vector3 pos = transform.position;
        const float magnitude = 5;
        for (int i = 0; i < numberRays; i++)
        {
            Vector3 dir = Quaternion.Euler(0, i * delta, 0) * transform.right;
            if (Physics.Raycast(pos, dir, out hit, magnitude))
            {
                if (hit.collider.CompareTag("Wall"))
                    sensor.AddObservation(new Vector3(0, 0, 1));
                else if (hit.collider.CompareTag("Target"))
                    sensor.AddObservation(new Vector3(0, 1, 0));
                else
                    Debug.LogError("Unresolved tag " + hit.collider.tag);

            }
            else
            {
                sensor.AddObservation(Vector3.zero);
            }
        }

        sensor.AddObservation(1f / (currentRoom.timesRight + 1));
        sensor.AddObservation(1f / (currentRoom.timesDown + 1));
        sensor.AddObservation(1f / (currentRoom.timesLeft + 1));
        sensor.AddObservation(1f / (currentRoom.timesUp + 1));

        if (currentRoom.right != null)
            sensor.AddObservation(1f / (currentRoom.right.timesVisited + 1));
        else
            sensor.AddObservation(0);

        if (currentRoom.down != null)
            sensor.AddObservation(1f / (currentRoom.down.timesVisited + 1));
        else
            sensor.AddObservation(0);

        if (currentRoom.left != null)
            sensor.AddObservation(1f / (currentRoom.left.timesVisited + 1));
        else
            sensor.AddObservation(0);

        if (currentRoom.up != null)
            sensor.AddObservation(1f / (currentRoom.up.timesVisited + 1));
        else
            sensor.AddObservation(0);

    }
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // Action
        steps++;
        int action = actionBuffers.DiscreteActions[0];
        switch (action)
        {
            case 0: // left
                if (currentRoom.left != null)
                {
                    AddReward(-10 * (currentRoom.timesLeft + currentRoom.left.timesVisited) / (maze.roomsCount * 2));
                    currentRoom.timesLeft++;
                    currentRoom.left.timesVisited++;
                    transform.Translate(new Vector3(-5f, 0, 0));
                    currentRoom = currentRoom.left;
                }
                else
                {
                    AddReward(-100 / (maze.roomsCount * 2));
                }
                break;
            case 1: // right
                if (currentRoom.right != null)
                {
                    AddReward(-10 * (currentRoom.timesRight + currentRoom.right.timesVisited) / (maze.roomsCount * 2));
                    currentRoom.timesRight++;
                    currentRoom.right.timesVisited++;
                    transform.Translate(new Vector3(5f, 0, 0));
                    currentRoom = currentRoom.right;
                }
                else
                {
                    AddReward(-100 / (maze.roomsCount * 2));
                }
                break;
            case 2: // up
                if (currentRoom.up != null)
                {
                    AddReward(-10 * (currentRoom.timesUp + currentRoom.up.timesVisited) / (maze.roomsCount * 2));
                    currentRoom.timesUp++;
                    currentRoom.up.timesVisited++;
                    transform.Translate(new Vector3(0, 0, 5f));
                    currentRoom = currentRoom.up;
                }
                else
                {
                    AddReward(-100 / (maze.roomsCount * 2));
                }
                break;
            case 3: // down
                if (currentRoom.down != null)
                {
                    AddReward(-10 * (currentRoom.timesDown + currentRoom.down.timesVisited) / (maze.roomsCount * 2));
                    currentRoom.timesDown++;
                    currentRoom.down.timesVisited++;
                    transform.Translate(new Vector3(0, 0, -5f));
                    currentRoom = currentRoom.down;
                }
                else
                {
                    AddReward(-100 / (maze.roomsCount * 2));
                }
                break;
            default:
                break;
        }

        // Reward
        if (Vector3.Distance(transform.position, maze.target.position) < 2)
        {
            SetReward(100);
            maze.solvedTimes++;
            EndEpisode();
        }
        else if (steps > maze.roomsCount * 2)
        {
            steps = 0;
            EndEpisode();
        }
        else if (GetCumulativeReward() < -100)
        {
            steps = 0;
            EndEpisode();
        }
    }
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        if (Input.GetKeyDown(KeyCode.D))
            discreteActionsOut[0] = 1;
        else if (Input.GetKeyDown(KeyCode.A))
            discreteActionsOut[0] = 0;
        else if (Input.GetKeyDown(KeyCode.S))
            discreteActionsOut[0] = 3;
        else if (Input.GetKeyDown(KeyCode.W))
            discreteActionsOut[0] = 2;
    }
}
