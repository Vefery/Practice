using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

public enum MazeDirection
{
    left, right, up, down
}
public class Maze : MonoBehaviour
{
    [System.NonSerialized]
    public Transform target;
    public int depth = 10;
    public int depthIncrementStep;
    public int seed = -1;
    public int optimalSteps = 0;
    public bool clampSize = true;
    public bool placeSigns = false;
    public bool goalSpawned = false;
    public GameObject roomPrefab, wallPrefab, targetPrefab;
    public GameObject leftSignPrefab, upSignPrefab, rightSignPrefab, downSignPrefab;
    [System.NonSerialized]
    public Vector3 startPose;
    [System.NonSerialized]
    public List<Vector3> tilePositions = new();
    public List<Room> rooms = new();
    public int initDepth;
    public int solvedTimes = 1;
    [System.NonSerialized]
    public List<Action> roomsCreateActions = new();
    [System.NonSerialized]
    public Room targetHolder;
    public int roomsCount {  get { return rooms.Count; } }

    public event Action FinishEvent;

    private void Awake()
    {
        startPose = transform.position;
        tilePositions.Add(startPose);
        initDepth = depth;
        rooms.Add(Instantiate(roomPrefab, transform).GetComponent<Room>());
    }

    private void Start()
    {
        //GenerateMaze();
    }
    public void GenerateMaze()
    {
        if (seed != -1)
            UnityEngine.Random.InitState(seed);
        rooms[0].Generate(this, UnityEngine.Random.state);
        rooms[0].neighboursCount--;
        for (int i = 0; i < roomsCreateActions.Count; i++)
        {
            roomsCreateActions[i]();
        }
        if (!goalSpawned)
        {
            goalSpawned = true;
            target = Instantiate(targetPrefab, rooms.Last().transform.position, Quaternion.identity, rooms.Last().transform).transform;
            targetHolder = rooms.Last();
        }
        roomsCreateActions.Clear();

        Room prevRoom;
        Room curRoom = targetHolder;
        int protector = 1000;
        while (curRoom != rooms[0] && protector > 0)
        {
            protector--;
            prevRoom = curRoom;
            curRoom = curRoom.transform.parent.GetComponent<Room>();
            optimalSteps++;            
        }

        if (placeSigns)
        {
            curRoom = targetHolder;
            protector = 1000;
            while (curRoom != rooms[0] && protector > 0)
            {
                protector--;
                prevRoom = curRoom;
                curRoom = curRoom.transform.parent.GetComponent<Room>();
                if (curRoom.neighboursCount > 2)
                {
                    if (prevRoom == curRoom.left)
                        Instantiate(leftSignPrefab, curRoom.transform.position, Quaternion.identity, curRoom.transform);
                    if (prevRoom == curRoom.up)
                        Instantiate(upSignPrefab, curRoom.transform.position, Quaternion.identity, curRoom.transform);
                    if (prevRoom == curRoom.right)
                        Instantiate(rightSignPrefab, curRoom.transform.position, Quaternion.identity, curRoom.transform);
                    if (prevRoom == curRoom.down)
                        Instantiate(downSignPrefab, curRoom.transform.position, Quaternion.identity, curRoom.transform);
                }
            }
        }
    }
    public void ResetMaze()
    {
        if (!clampSize)
            initDepth++;
        else if (solvedTimes % depthIncrementStep == 0 && initDepth < 20)
            initDepth++;
        FinishEvent?.Invoke();
        depth = initDepth;
        goalSpawned = false;
        tilePositions.Clear();
        foreach (Room room in rooms)
            Destroy(room.gameObject);
        rooms.Clear();
        rooms.Add(Instantiate(roomPrefab, transform).GetComponent<Room>());
        tilePositions.Add(rooms[0].transform.position);
        GenerateMaze();
    }
    public bool CheckPosition(Vector3 pos)
    {
        foreach (Vector3 p in tilePositions)
        {
            if (Vector3.Distance(p, pos) < 5)
                return false;
        }
        return true;
    }
}
