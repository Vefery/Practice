using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.InputSystem;

[RequireComponent(typeof(CapsuleCollider))]
public class DeterministicAgent : MonoBehaviour
{
    public float totalAngle = 360;
    public float numberRays = 32;
    private float delta;
    private RaycastHit hit;
    private CapsuleCollider capsuleCollider;
    private Maze maze;
    public int steps = 0;
    public Room currentRoom;

    private void Start()
    {
        delta = totalAngle / numberRays;
        maze = FindFirstObjectByType<Maze>();
        capsuleCollider = GetComponent<CapsuleCollider>();

        RestartEpisode();
    }

    private void Update()
    {
        ExecuteAction(ChooseAction(CollectObservations()));

        Vector3 pos = transform.position;
        const float magnitude = 5;

        for (int i = 0; i < numberRays; i++)
        {
            Vector3 dir = Quaternion.Euler(0, i * delta, 0) * transform.right;
            Debug.DrawLine(pos, pos + dir * magnitude);
        }
    }

    public void RestartEpisode()
    {
        Vector3 startPos = maze.startPose;

        maze.ResetMaze();
        steps = 0;
        currentRoom = maze.rooms[0];

        transform.position = startPos + new Vector3(0, capsuleCollider.height / 2, 0);
    }
    public List<float> CollectObservations()
    {
        List<float> obs = new();
        Vector3 pos = transform.position;
        const float magnitude = 5;
        for (int i = 0; i < numberRays; i++)
        {
            Vector3 dir = Quaternion.Euler(0, i * delta, 0) * transform.right;
            if (Physics.Raycast(pos, dir, out hit, magnitude))
            {
                if (hit.collider.CompareTag("Wall"))
                    obs.Add(0);
                else if (hit.collider.CompareTag("Target"))
                    obs.Add(1);
                else
                    Debug.LogError("Unresolved tag " + hit.collider.tag);
            }
            else
            {
                obs.Add(-1);
            }
        }

        obs.Add(currentRoom.timesRight);
        obs.Add(currentRoom.timesDown);
        obs.Add(currentRoom.timesLeft);
        obs.Add(currentRoom.timesUp);

        if (currentRoom.right != null)
            obs.Add(currentRoom.right.timesVisited);
        else
            obs.Add(100);

        if (currentRoom.down != null)
            obs.Add(currentRoom.down.timesVisited);
        else
            obs.Add(100);

        if (currentRoom.left != null)
            obs.Add(currentRoom.left.timesVisited);
        else
            obs.Add(100);

        if (currentRoom.up != null)
            obs.Add(currentRoom.up.timesVisited);
        else
            obs.Add(100);

        return obs;
    }
    public int ChooseAction(List<float> obs)
    {
        List<(string, float, float, float)> temp = new() { ("Right", obs[0], obs[4], obs[8]), ("Down", obs[1], obs[5], obs[9]), ("Left", obs[2], obs[6], obs[10]), ("Up", obs[3], obs[7], obs[11]) };

        var optimal = temp.Find(x => x.Item2 == 1);

        if (optimal.Item1 == null)
        {
            temp = temp.Where(x => x.Item2 != 0).ToList();
            optimal = temp.OrderBy(x => x.Item3).ThenBy(x => x.Item4).First();
        }
        switch (optimal.Item1)
        {
            case "Right":
                return 1;
            case "Down":
                return 3;
            case "Left":
                return 0;
            case "Up":
                return 2;
            default:
                return -1;
        }
    }
    public void ExecuteAction(int action)
    {
        // Action
        steps++;
        switch (action)
        {
            case 0: // left
                currentRoom.timesLeft++;
                currentRoom.left.timesVisited++;
                transform.Translate(new Vector3(-5f, 0, 0));
                currentRoom = currentRoom.left;
                break;
            case 1: // right
                currentRoom.timesRight++;
                currentRoom.right.timesVisited++;
                transform.Translate(new Vector3(5f, 0, 0));
                currentRoom = currentRoom.right;
                break;
            case 2: // up
                currentRoom.timesUp++;
                currentRoom.up.timesVisited++;
                transform.Translate(new Vector3(0, 0, 5f));
                currentRoom = currentRoom.up;
                break;
            case 3: // down
                currentRoom.timesDown++;
                currentRoom.down.timesVisited++;
                transform.Translate(new Vector3(0, 0, -5f));
                currentRoom = currentRoom.down;
                break;
            default:
                break;
        }

        // Reward
        if (Vector3.Distance(transform.position, maze.target.position) < 2)
        {
            maze.solvedTimes++;
            Debug.Log("Succeeded");
            RestartEpisode();
        }
        else if (steps > maze.roomsCount * 2)
        {
            Debug.Log("Failed");
            steps = 0;
            RestartEpisode();
        }
    }
}
